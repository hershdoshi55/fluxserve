# app/batch_worker.py
import asyncio
import torch
import time
from typing import List, Optional
from dataclasses import dataclass, field
from app.queue_manager import QueueManager, InferenceRequest
from app.kv_cache import KVCachePool
from app.metrics import (
    batch_size_hist, active_sequences_gauge,
    tokens_per_second_gauge, queue_depth_gauge, gpu_memory_used_gauge
)

@dataclass
class ActiveSequence:
    request: InferenceRequest
    input_ids: torch.Tensor          # Current token sequence on GPU
    attention_mask: torch.Tensor     # Attention mask
    past_key_values: Optional[tuple] # KV cache for this sequence
    tokens_generated: int = 0
    start_time: float = field(default_factory=time.time)
    first_token_time: float = 0.0   # seconds elapsed from start to first token

class BatchWorker:
    def __init__(
        self,
        model,
        tokenizer,
        queue_manager: QueueManager,
        kv_cache_pool: KVCachePool,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.queue = queue_manager
        self.kv_cache = kv_cache_pool
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.active: List[ActiveSequence] = []
        self._running = False
        self._token_count = 0
        self._last_throughput_calc = time.time()

    async def run(self):
        """Main loop — runs forever as an asyncio task."""
        self._running = True
        while self._running:
            # Pull new requests into the batch
            await self._fill_batch()

            if not self.active:
                # Nothing to do — yield control and try again
                await asyncio.sleep(0.001)
                continue

            # Run one decode step for all active sequences
            await asyncio.get_event_loop().run_in_executor(
                None, self._decode_step
            )

            # Check for finished sequences and resolve their futures
            self._resolve_finished()

            # Update metrics
            self._update_metrics()

    async def _fill_batch(self):
        """Pull waiting requests into the active batch."""
        slots_available = self.max_batch_size - len(self.active)
        if slots_available == 0:
            return

        deadline = time.time() + (self.max_wait_ms / 1000)

        while len(self.active) < self.max_batch_size:
            try:
                timeout = max(0, deadline - time.time())
                request = await asyncio.wait_for(
                    self.queue.dequeue(), timeout=timeout
                )
                # Run prefill for this new sequence
                seq = await asyncio.get_event_loop().run_in_executor(
                    None, self._prefill, request
                )
                self.active.append(seq)
            except asyncio.TimeoutError:
                break

    def _prefill(self, request: InferenceRequest) -> ActiveSequence:
        """
        Run the prefill step for a new request.
        This processes the entire prompt and populates the KV cache.
        """
        from app.model_loader import SYSTEM_PROMPT
        prefill_start = time.time()
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{request.text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # Forward pass with use_cache=True to get past_key_values
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True
            )

        first_token_time = time.time() - prefill_start

        # The model's last token logit tells us the first generated token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Build the sequence with the first generated token appended
        new_input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
        new_attention_mask = torch.cat([
            inputs.attention_mask,
            torch.ones((1, 1), device="cuda")
        ], dim=-1)

        return ActiveSequence(
            request=request,
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            past_key_values=outputs.past_key_values,
            tokens_generated=1,
            first_token_time=first_token_time,
        )

    def _decode_step(self):
        """
        Run one decode step for ALL active sequences simultaneously.
        This is the continuous batching core — all sequences advance together.

        Note: True continuous batching requires padding handling and variable-length
        sequences. This implementation runs sequences independently per step for
        clarity. A production implementation would pad and batch the decode calls.
        """
        for seq in self.active:
            with torch.no_grad():
                # Only pass the LAST token — KV cache handles the rest
                last_token = seq.input_ids[:, -1:]
                last_mask = seq.attention_mask

                outputs = self.model(
                    input_ids=last_token,
                    attention_mask=last_mask,
                    past_key_values=seq.past_key_values,
                    use_cache=True,
                    return_dict=True
                )

            # Sample next token (greedy)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            # Update sequence state
            seq.input_ids = torch.cat([seq.input_ids, next_token], dim=-1)
            seq.attention_mask = torch.cat([
                seq.attention_mask,
                torch.ones((1, 1), device="cuda")
            ], dim=-1)
            seq.past_key_values = outputs.past_key_values
            seq.tokens_generated += 1
            self._token_count += 1

    def _resolve_finished(self):
        """
        Check which sequences are done and resolve their HTTP futures.
        A sequence is done if it generated a stop token or hit max_new_tokens.
        """
        still_active = []
        for seq in self.active:
            last_token_id = seq.input_ids[0, -1].item()
            is_eos = last_token_id == self.tokenizer.eos_token_id
            is_max = seq.tokens_generated >= seq.request.max_new_tokens

            if is_eos or is_max:
                # Decode the generated portion
                generated_ids = seq.input_ids[0, -(seq.tokens_generated):]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip().lower()

                label = "toxic" if "toxic" in generated_text else \
                        "safe" if "safe" in generated_text else "unknown"

                latency_ms = (time.time() - seq.start_time) * 1000
                ttft_ms = seq.first_token_time * 1000  # set during prefill
                tpot_ms = (latency_ms - ttft_ms) / max(seq.tokens_generated - 1, 1)

                result = {
                    "label": label,
                    "flagged": label == "toxic",
                    "tokens_generated": seq.tokens_generated,
                    "latency_ms": latency_ms,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                }

                # Resolve the Future — this wakes up the HTTP handler
                if not seq.request.future.done():
                    seq.request.future.set_result(result)
            else:
                still_active.append(seq)

        self.active = still_active

    def _update_metrics(self):
        batch_size_hist.observe(len(self.active))
        active_sequences_gauge.set(len(self.active))
        queue_depth_gauge.set(self.queue.depth())

        # Calculate tokens/sec
        now = time.time()
        elapsed = now - self._last_throughput_calc
        if elapsed >= 1.0:
            tokens_per_second_gauge.set(self._token_count / elapsed)
            self._token_count = 0
            self._last_throughput_calc = now

        # GPU memory
        try:
            import torch
            gpu_memory_used_gauge.set(torch.cuda.memory_allocated() / 1e9)
        except Exception:
            pass
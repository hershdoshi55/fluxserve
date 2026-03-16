# scripts/benchmark.py
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_TEXT = "I hate everything about this. You should all be ashamed."
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50
MAX_NEW_TOKENS = 10

SYSTEM_PROMPT = (
    "You are a content moderation classifier. "
    "Reply with exactly one word: 'toxic' or 'safe'."
)

def build_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

print("Loading model (FP16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="cuda"
)
model.eval()
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

inputs = tokenizer(build_prompt(TEST_TEXT), return_tensors="pt").to("cuda")
latencies = []
tokens_generated = []

print(f"\nWarmup ({WARMUP_RUNS} runs)...")
for _ in range(WARMUP_RUNS):
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                       do_sample=False, pad_token_id=tokenizer.eos_token_id)

print(f"Benchmarking ({BENCHMARK_RUNS} runs)...")
for _ in range(BENCHMARK_RUNS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)
    output_ids = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences  # type: ignore[union-attr]
    tokens_generated.append(output_ids.shape[1] - inputs.input_ids.shape[1])

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
p99 = latencies[int(len(latencies) * 0.99)]
avg = sum(latencies) / len(latencies)
avg_tokens = sum(tokens_generated) / len(tokens_generated)

print(f"\n{'='*50}")
print(f"FP16 Single-Request Benchmark (sequential, no server)")
print(f"{'='*50}")
print(f"VRAM used:        {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Avg tokens/req:   {avg_tokens:.1f}")
print(f"Latency p50:      {p50:.1f} ms")
print(f"Latency p95:      {p95:.1f} ms")
print(f"Latency p99:      {p99:.1f} ms")
print(f"Latency avg:      {avg:.1f} ms")
print(f"Throughput:       {1000/avg:.2f} req/sec")
print(f"Tokens/sec:       {avg_tokens * 1000/avg:.2f}")
print(f"{'='*50}")
print("\nRecord these numbers in benchmarks/phase6_single_request.md")
print("These are your pre-batching baseline for the Phase 7 comparison.")
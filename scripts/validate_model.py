# scripts/validate_model.py
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a content moderation classifier. "
    "Given a piece of text, determine if it is toxic or safe. "
    "Toxic content includes hate speech, threats, harassment, and explicit harmful content. "
    "Reply with exactly one word: 'toxic' or 'safe'."
)

test_cases = [
    ("I love this product, it is amazing!", "safe"),
    ("I will hurt you if you do not comply", "toxic"),
    ("The weather is nice today", "safe"),
    ("You are worthless and should disappear", "toxic"),
    ("This is a great community!", "safe"),
    ("Go kill yourself", "toxic"),
]


def classify(model, tokenizer, text: str) -> tuple[str, float]:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = (time.time() - start) * 1000
    generated = str(tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )).strip().lower()
    label = "toxic" if "toxic" in generated else "safe" if "safe" in generated else "unknown"
    return label, latency


def run_validation(model, tokenizer, precision_label: str) -> dict:
    print(f"\nRunning validation ({precision_label})...")
    correct = 0
    latencies = []
    for text, expected in test_cases:
        label, latency = classify(model, tokenizer, text)
        latencies.append(latency)
        ok = label == expected
        correct += ok
        status = "✓" if ok else "✗"
        print(f"  {status} [{latency:.0f}ms] Expected: {expected:5s} Got: {label:7s} | {text[:60]}")
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\n  Accuracy: {correct}/{len(test_cases)}")
    print(f"  VRAM after inference: {vram:.2f} GB")
    return {
        "precision": precision_label,
        "accuracy": f"{correct}/{len(test_cases)}",
        "avg_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "vram_gb": vram,
    }


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- FP16 ---
print("\nLoading model (FP16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
fp16_results = run_validation(model, tokenizer, "FP16")

# Unload FP16 before loading INT8
del model
gc.collect()
torch.cuda.empty_cache()

# --- INT8 ---
print("\nLoading model (INT8)...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda"
)
model.eval()
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
int8_results = run_validation(model, tokenizer, "INT8")

# --- Comparison ---
print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'FP16':>15} {'INT8':>15}")
print("-" * 60)
print(f"{'Accuracy':<25} {fp16_results['accuracy']:>15} {int8_results['accuracy']:>15}")
print(f"{'Avg latency (ms)':<25} {fp16_results['avg_latency_ms']:>15.0f} {int8_results['avg_latency_ms']:>15.0f}")
print(f"{'Min latency (ms)':<25} {fp16_results['min_latency_ms']:>15.0f} {int8_results['min_latency_ms']:>15.0f}")
print(f"{'Max latency (ms)':<25} {fp16_results['max_latency_ms']:>15.0f} {int8_results['max_latency_ms']:>15.0f}")
print(f"{'VRAM (GB)':<25} {fp16_results['vram_gb']:>15.2f} {int8_results['vram_gb']:>15.2f}")
print("=" * 60)

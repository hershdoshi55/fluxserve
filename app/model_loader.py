# app/model_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer

SYSTEM_PROMPT = (
    "You are a content moderation classifier. "
    "Reply with exactly one word: 'toxic' or 'safe'."
)

def run_inference(model, tokenizer, text: str, max_new_tokens: int = 10) -> dict:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip().lower()
    label = "toxic" if "toxic" in generated else "safe" if "safe" in generated else "unknown"
    return {"label": label, "flagged": label == "toxic"}
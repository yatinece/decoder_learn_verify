import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from Tokenizer.BpeToken import BpeTokenizer
from config.ConfigFile import Config
from transformer.transformer import *

# -------------------------------
# Config and setup
# -------------------------------
config = Config()
print(dict(config.__dict__.items()))

sample_method = "sample"   # "argmax" or "sample"
temperature = 1.2          # <1.0 makes output more deterministic
top_k = 50                 # keep only top K tokens
top_p = 0.6                # nucleus sampling
max_new_tokens = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Load model and tokenizer
# -------------------------------
model = torch.load(os.path.join(config.data_path, "best_model_full.pt"), weights_only=False)
model.to(device)
model.eval()

tokenizer = BpeTokenizer.load(config.path)
eos_token_id = tokenizer.tokenizer.token_to_id("<EOS>")

print("Model and tokenizer loaded.")
time.sleep(1)

# -------------------------------
# Sampling helpers
# -------------------------------
def apply_temperature_and_filtering(logits, temperature=1.0, top_k=0, top_p=1.0):
    # scale by temperature
    logits = logits / max(temperature, 1e-8)

    # top-k
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits[logits < min_values] = -float("Inf")

    # top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # remove tokens with cumulative prob above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float("Inf")

    return logits

# -------------------------------
# Interactive loop
# -------------------------------
while True:
    print("\nEnter Text (or 'exit' to quit):")
    input_text = input().strip()
    if input_text.lower() == "exit":
        break

    token_data = tokenizer.encode(input_text)
    print(f"Input '{input_text}' → tokens {token_data}")

    if eos_token_id is not None and token_data and token_data[-1] == eos_token_id:
        token_data = token_data[:-1]

    # start generating
    print("Start inference...")
    with torch.no_grad():
        for step in tqdm(range(max_new_tokens), desc="Generating Tokens", total=max_new_tokens):
            token_tensor_list_nopad = convert_listbatch_to_listtensor(
                [token_data], device, max_length=model.max_length, dtype=torch.long
            )
            if len(token_data) < model.max_length:
                token_tensor_list = pad_sequence_list(token_tensor_list_nopad, model.max_length, pad_token_id=0)
            else:
                token_tensor_list = token_tensor_list_nopad[:, -model.max_length:]

            output, _ = model(token_tensor_list)
            logits = output[:, -1, :]  # last token logits

            if sample_method == "argmax":
                next_token = torch.argmax(logits, dim=-1)
            else:
                # apply temperature + top-k/top-p filtering
                logits = apply_temperature_and_filtering(logits, temperature, top_k, top_p)
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            next_token_id = next_token.item()
            token_data.append(next_token_id)

            # incremental decode
            partial_text = tokenizer.decode(token_data, skip_special_tokens=True).rstrip()
            print(f"Step {step+1}: {partial_text}")

            if eos_token_id is not None and next_token_id == eos_token_id:
                print("EOS token generated. Stopping early.")
                break

    # Final decode
    output = tokenizer.decode(token_data, skip_special_tokens=True).rstrip()
    print(f"\nFinal Output: {output}")

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformer.transformer2 import *
from Tokenizer.BpeToken import BpeTokenizer
from config.ConfigFile import Config
from transformer.transformer2 import convert_listbatch_to_listtensor, pad_sequence_list

# -------------------------------
# Config and setup
# -------------------------------
config = Config()

sample_method = "sample"   # "argmax" or "sample"
temperature = 0.8          # <1.0 makes output more deterministic
top_k = 50                 # keep only top K tokens
top_p = 0.9                # nucleus sampling
max_new_tokens = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------
# Load model and tokenizer
# -------------------------------
print("Loading model and tokenizer...")
model = torch.load(os.path.join(config.data_path, "best_model_full.pt"), weights_only=False)
model.to(device)
model.eval()

tokenizer = BpeTokenizer.load(config.path)
eos_token_id = tokenizer.tokenizer.token_to_id("<EOS>")
bos_token_id = tokenizer.tokenizer.token_to_id("<BOS>")
pad_token_id = tokenizer.tokenizer.token_to_id("<PAD>")

print(f"Model loaded. Vocab size: {model.vocab_size}, Max length: {model.max_length}")
print(f"Special tokens - BOS: {bos_token_id}, EOS: {eos_token_id}, PAD: {pad_token_id}")

# -------------------------------
# Sampling helpers
# -------------------------------
def apply_temperature_and_filtering(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Apply temperature scaling and top-k/top-p filtering to logits.
    Args:
        logits: [batch_size, vocab_size]
    Returns:
        filtered_logits: [batch_size, vocab_size]
    """
    # Scale by temperature
    logits = logits / max(temperature, 1e-8)

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter back to original indexing
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float('inf')

    return logits

def generate_text(prompt, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.9, sample_method="sample"):
    """
    Generate text given a prompt.
    """
    # Encode input
    token_data = tokenizer.encode(prompt)
    print(f"\nInput: '{prompt}'")
    print(f"Tokens: {token_data} (length: {len(token_data)})")
    
    # Remove EOS if present at end
    if eos_token_id is not None and token_data and token_data[-1] == eos_token_id:
        token_data = token_data[:-1]
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Prepare input: take last max_length tokens if sequence is too long
            current_tokens = token_data[-model.max_length:] if len(token_data) > model.max_length else token_data
            
            # Convert to tensor and pad
            token_tensor_list = convert_listbatch_to_listtensor(
                [current_tokens], device, max_length=model.max_length, dtype=torch.long
            )
            token_tensor = pad_sequence_list(token_tensor_list, model.max_length, pad_token_id=0)
            
            # Forward pass
            output, _ = model(token_tensor)
            
            # Get logits for the last real token (not padding)
            last_token_idx = min(len(current_tokens) - 1, model.max_length - 1)
            logits = output[:, last_token_idx, :]
            
            # Sample next token
            if sample_method == "argmax":
                next_token = torch.argmax(logits, dim=-1)
            else:
                # Apply temperature and filtering
                filtered_logits = apply_temperature_and_filtering(logits, temperature, top_k, top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            next_token_id = next_token.item()
            
            # Stop if EOS generated
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            
            # Append to sequences
            token_data.append(next_token_id)
            generated_tokens.append(next_token_id)
    
    # Decode full sequence
    full_output = tokenizer.decode(token_data, skip_special_tokens=True)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return full_output, generated_text, len(generated_tokens)

# -------------------------------
# Interactive loop
# -------------------------------
print("\n" + "="*60)
print("Text Generation Interface")
print("="*60)
print(f"Settings: method={sample_method}, temp={temperature}, top_k={top_k}, top_p={top_p}")
print("="*60)

while True:
    print("\n" + "-"*60)
    print("Enter your prompt (or 'exit' to quit, 'settings' to change parameters):")
    user_input = input("> ").strip()
    
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    if user_input.lower() == "settings":
        print(f"\nCurrent settings:")
        print(f"  sample_method: {sample_method}")
        print(f"  temperature: {temperature}")
        print(f"  top_k: {top_k}")
        print(f"  top_p: {top_p}")
        print(f"  max_new_tokens: {max_new_tokens}")
        
        try:
            new_temp = input(f"Temperature [{temperature}]: ").strip()
            if new_temp:
                temperature = float(new_temp)
            
            new_top_k = input(f"Top-k [{top_k}]: ").strip()
            if new_top_k:
                top_k = int(new_top_k)
            
            new_top_p = input(f"Top-p [{top_p}]: ").strip()
            if new_top_p:
                top_p = float(new_top_p)
            
            new_max = input(f"Max tokens [{max_new_tokens}]: ").strip()
            if new_max:
                max_new_tokens = int(new_max)
            
            print("Settings updated!")
        except ValueError:
            print("Invalid input. Settings unchanged.")
        continue
    
    if not user_input:
        print("Please enter a prompt.")
        continue
    
    # Generate
    print("\nGenerating...")
    full_output, generated_only, num_tokens = generate_text(
        user_input,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_method=sample_method
    )
    
    print("\n" + "="*60)
    print("GENERATED OUTPUT")
    print("="*60)
    print(full_output)
    print("="*60)
    print(f"Generated {num_tokens} new tokens")
    print("="*60)
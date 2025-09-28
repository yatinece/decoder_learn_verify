import time
from datasets import load_dataset
from Tokenizer.CharToken import CharTokenizer
from Tokenizer.BpeToken import BpeTokenizer

# ------------------------------
# Benchmark with WikiText-2
# ------------------------------
start_total = time.time()  # start total timer

#  Load WikiText-2 (raw)
start = time.time()
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]
print(f"Loaded dataset in {time.time() - start:.2f} seconds")

# Remove empty lines
train_texts = [line for line in train_texts if line.strip() != ""]

# ------------------------------
# CharTokenizer
# ------------------------------
print("\n=== CharTokenizer ===")
start = time.time()
char_tokenizer = CharTokenizer(train_texts)
print(f"Tokenizer initialized in {time.time() - start:.2f} seconds")

start = time.time()
encoded_dataset_char = [
    char_tokenizer.encode(line, add_bos=True, add_eos=True) 
    for line in train_texts
]
print(f"Encoding dataset completed in {time.time() - start:.2f} seconds")

print("Number of lines:", len(encoded_dataset_char))
print("First encoded line IDs:", encoded_dataset_char[0])
print("Decoded line:", char_tokenizer.decode(encoded_dataset_char[0], show_special_tokens=True))

# ------------------------------
# BpeTokenizer
# ------------------------------
print("\n=== BpeTokenizer ===")
start = time.time()
bpe_tokenizer = BpeTokenizer(train_texts, vocab_size=5000)  # can tune vocab_size
print(f"Tokenizer initialized in {time.time() - start:.2f} seconds")

start = time.time()
encoded_dataset_bpe = [
    bpe_tokenizer.encode(line)  # BOS/EOS are auto added via TemplateProcessing
    for line in train_texts
]
print(f"Encoding dataset completed in {time.time() - start:.2f} seconds")

print("Number of lines:", len(encoded_dataset_bpe))
print("First encoded line IDs:", encoded_dataset_bpe[0])
print("Decoded line:", bpe_tokenizer.decode(encoded_dataset_bpe[0], skip_special_tokens=False))

print(f"\nTotal time: {time.time() - start_total:.2f} seconds")

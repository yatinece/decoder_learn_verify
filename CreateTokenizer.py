from Tokenizer.BpeToken import BpeTokenizer
from config.ConfigFile import Config
from datasets import load_dataset
import argparse
import time
import torch
import numpy as np
import os
from tqdm import tqdm

def load_dataset_all(download_data_key,datasets="wikitext"):

    if  download_data_key[datasets][1] is not None :
        dataset = load_dataset(download_data_key[datasets][0], download_data_key[datasets][1])
    else:
        dataset = load_dataset(download_data_key[datasets][0])

    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    #train_texts = [line for line in dataset[split_name]["text"] if line.strip() != ""]
    texts = np.array(dataset[split_name]["text"])
    mask = np.char.strip(texts) != ""
    train_texts = texts[mask].tolist()
    #train_texts = dataset[split_name]["text"]
    return train_texts

if __name__ == "__main__":
    ## Config files
    config_parser=Config()
    download_data_key = config_parser.download_data_key
    dataset = config_parser.dataset
    path = config_parser.path
    data_path= config_parser.data_path
    max_length = config_parser.max_length  # From your training config; adjust as needed
    stride = int(max_length * 0.9)  # 10% overlap; tune 0.8-0.95 for more/less density
    os.makedirs(data_path ,exist_ok=True)
    ## Load data sets
    train_texts = load_dataset_all(download_data_key,dataset)

    print(train_texts[:5])

    ## config BpeTokenizer
    #bpetoken = BpeTokenizer(train_texts)

    ## Save tokenizer
    #bpetoken.save(path)

    ## Load tokenizer 
    bpe2 = BpeTokenizer.load(path)

    # Test encode-decode
    sample = "Hello, world!"
    ids = bpe2.encode(sample)
    decoded = bpe2.decode(ids)

    print("Original:", sample)
    print("Token IDs:", ids)
    print("Decoded:", decoded)


    ## encode the data in batch 
    batch_size_token = config_parser.batch_size_token
    encoded_dataset = []
    start_time = time.time()
    for k in tqdm(range(0, len(train_texts), batch_size_token), desc="Encoding batches"):
        batch = train_texts[k:k+batch_size_token]
        batch_encoded = bpe2.tokenizer.encode_batch(batch)
        batch_ids = [e.ids for e in batch_encoded]
        encoded_dataset.extend(batch_ids)
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} sec | {elapsed/60:.2f} min | {elapsed/3600:.2f} hr")
    start_time = time.time()
    torch.save(encoded_dataset, data_path+"encoded_dataset.pt")
    elapsed = time.time() - start_time
    print(f"Save time: {elapsed:.2f} sec | {elapsed/60:.2f} min | {elapsed/3600:.2f} hr")
    print(f"few rows {encoded_dataset[:3]}")
    ### Test Fast Save
    start_time = time.time()
    for k in tqdm(range(0, len(train_texts), batch_size_token), desc="Encoding batches"):
        batch = train_texts[k:k+batch_size_token]
        batch_encoded = bpe2.tokenizer.encode_batch_fast(batch)
        batch_ids = [e.ids for e in batch_encoded]
        encoded_dataset.extend(batch_ids)
    elapsed = time.time() - start_time
    print(f"Elapsed time Fast: {elapsed:.2f} sec | {elapsed/60:.2f} min | {elapsed/3600:.2f} hr")
    start_time = time.time()
    torch.save(encoded_dataset, data_path+"encoded_dataset_fast.pt")
    elapsed = time.time() - start_time
    print(f"Save time Fast: {elapsed:.2f} sec | {elapsed/60:.2f} min | {elapsed/3600:.2f} hr")
    print(f"few rows {encoded_dataset[:3]}")


    print(f"full text of all concatenated")
    separator = '\n\n'  # Or ' ' if lines are sentences; preserves article structure
    all_data = separator.join([t.strip() for t in train_texts if t.strip()])
    print(f"few rows train_texts :  {train_texts[:3]}")
    print(f"few rows all_data : {all_data[:300]}")

    print("Encoding full text...")
    start_time = time.time()
    full_encoded = bpe2.tokenizer.encode(all_data)  # Single encode; adds no BOS/EOS here (handle in chunks)
    torch.save(full_encoded, data_path + "full_encoded.pt")
    full_ids = full_encoded.ids
    torch.save(full_ids, data_path + "full_ids.pt")
    elapsed = time.time() - start_time
    print(f"Full encode time: {elapsed:.2f} sec")

    encoded_dataset = []

    for i in tqdm(range(0, len(full_ids) - max_length + 1, stride), desc="Chunking sequences"):
        chunk_ids = full_ids[i:i + max_length]
        if len(chunk_ids) < max_length * 0.5:  # Skip tiny tail
            break
        # Wrap with specials (mimic per-text)
        bos_id = bpe2.tokenizer.token_to_id("<BOS>")
        eos_id = bpe2.tokenizer.token_to_id("<EOS>")
        wrapped_chunk = [bos_id] + chunk_ids + [eos_id]
        encoded_dataset.append(wrapped_chunk[:max_length])  # Trim if over (rare)

    print(f"Created {len(encoded_dataset)} sequences (avg len: {np.mean([len(seq) for seq in encoded_dataset]):.0f})")

    # Save (use fast if available, but single encode is already fast)
    start_time = time.time()
    torch.save(encoded_dataset, data_path + "encoded_dataset_long.pt")
    elapsed = time.time() - start_time
    print(f"Save time: {elapsed:.2f} sec")
    print(f"Few rows: {encoded_dataset[:3]}")

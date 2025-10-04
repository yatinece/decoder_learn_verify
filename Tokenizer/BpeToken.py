from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json 
from datasets.arrow_dataset import Column
class BpeTokenizer:
    def __init__(self, texts=None, extra_tokens=None, vocab_size=16384):
        self.vocab_size = vocab_size
        self.special_tokens = extra_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        
        # Train if texts provided
        if texts is not None:
            self.tokenize(texts)
        
        # Set decoder
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <EOS> $B:1 <EOS>:1",
            special_tokens=[("<BOS>", self.tokenizer.token_to_id("<BOS>")),
                            ("<EOS>", self.tokenizer.token_to_id("<EOS>"))]
        )

    def convert_text(self, text):
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list):
            return text
        elif isinstance(text, Column):
            return list(text)
        else:
            raise ValueError(f"Not Implemented for: {type(text)}")

    def tokenize(self, texts):
        all_texts = self.convert_text(texts)
        self.tokenizer.train_from_iterator(all_texts, trainer=self.trainer)

    def encode(self, text):
        encoded = self.tokenizer.encode(text)
        return encoded.ids  # return list of token IDs

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)  # decode list of token IDs

    def save(self,path="./data/tokenizer.json"):
        self.tokenizer.save(path)
        print(f"Tokenizer saved at {path}")
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.tokenizer = Tokenizer.from_file(path)
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        obj.vocab_size = meta["vocab_size"]
        obj.special_tokens = meta["special_tokens"]
        return obj

# -------------------------------
# Example usage
if __name__ == "__main__":
    texts = "Hello, world!"
    
    bpe = BpeTokenizer(texts, vocab_size=100)
    
    token_ids = bpe.encode("Hello, world!")
    print("Token IDs:", token_ids)
    
    decoded_text = bpe.decode(token_ids)
    print("Decoded text:", decoded_text)

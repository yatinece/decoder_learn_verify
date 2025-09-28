class CharTokenizer:
    def __init__(self, texts=None , extra_tokens =None):
        self.dict_token_encoder = {}
        self.dict_token_decoder = {}
        self.extra_tokens = extra_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        if texts is not None:
            self.texts= self.convert_text(texts)
            self.texts = "".join(self.texts)
            self.tokenize()

    def convert_text(self,text):
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list):
            return text
        else:
            raise ValueError(f"Not Implemented for : {type(text)}")

    def tokenize(self, texts=None):
        if texts is None:
            text = self.texts
        else :
            text= "".join(self.convert_text(texts))
        
        unique_chars = list(set(text))
        self.dict_token_encoder = {char : i for i, char in enumerate(unique_chars)}
        self.dict_token_decoder = {i : char for i, char in enumerate(unique_chars)}

        offset=len(self.dict_token_encoder)
        for i, char in enumerate(self.extra_tokens):
            self.dict_token_encoder[char] =  i+offset
            self.dict_token_decoder[i+offset] = char

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = [self.dict_token_encoder.get(char,self.dict_token_encoder["<UNK>"]) for char in text]
        if add_bos :
            tokens = [self.dict_token_encoder["<BOS>"]] + tokens
        if add_eos:
            tokens = tokens + [self.dict_token_encoder["<EOS>"]]
        return tokens

    def decode(self, tokens, show_special_tokens=False):
        result = []
        for t in tokens:
            char = self.dict_token_decoder[t]
            if not show_special_tokens and char in self.extra_tokens:
                continue
            # replace special token with readable form if needed
            if show_special_tokens and char in self.extra_tokens:
                char = f" [{char}] "
            result.append(char)
        return "".join(result)


if __name__ == "__main__":
    texts = "Hello, World!"
    tokenizer = CharTokenizer(texts)

    token = tokenizer.encode(texts, add_bos=True, add_eos=True)
    print(token)
    print(tokenizer.decode(token,show_special_tokens=True))
    print(tokenizer.decode(token,show_special_tokens=False))
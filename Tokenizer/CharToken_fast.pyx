# Tokenizer/CharToken_fast.pyx
cimport cython

cdef class CharTokenizerC:
    cdef dict dict_token_encoder
    cdef dict dict_token_decoder
    cdef list extra_tokens

    def __init__(self, texts=None, extra_tokens=None):
        self.dict_token_encoder = {}
        self.dict_token_decoder = {}
        self.extra_tokens = extra_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        if texts is not None:
            cdef list all_texts = self.convert_text(texts)
            self.tokenize(all_texts)

    cpdef list convert_text(self, text):
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list):
            return text
        else:
            raise ValueError(f"Not Implemented for: {type(text)}")

    cpdef tokenize(self, list texts):
        cdef str combined = "".join(texts)
        cdef int offset, i
        cdef object c
        unique_chars = list(set(combined))

        for i, c in enumerate(unique_chars):
            self.dict_token_encoder[c] = i
            self.dict_token_decoder[i] = c

        offset = len(self.dict_token_encoder)
        for i, token in enumerate(self.extra_tokens):
            self.dict_token_encoder[token] = i + offset
            self.dict_token_decoder[i + offset] = token

    cpdef list encode(self, str text, bint add_bos=False, bint add_eos=False):
        cdef list tokens = [self.dict_token_encoder.get(c, self.dict_token_encoder["<UNK>"]) for c in text]
        if add_bos:
            tokens = [self.dict_token_encoder["<BOS>"]] + tokens
        if add_eos:
            tokens = tokens + [self.dict_token_encoder["<EOS>"]]
        return tokens

    cpdef str decode(self, list tokens, bint show_special_tokens=False):
        cdef list result = []
        cdef object char
        for char_id in tokens:
            char = self.dict_token_decoder[char_id]
            if not show_special_tokens and char in self.extra_tokens:
                continue
            if show_special_tokens and char in self.extra_tokens:
                char = f"[{char}]"
            result.append(char)
        return "".join(result)

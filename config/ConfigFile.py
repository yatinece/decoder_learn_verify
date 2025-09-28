class Config:
    def __init__(self):
        self.data_path = "./data/"
        self.path = "./data/tokenizer.json"
        self.dataset = "wikitext"
        self.batch_size_token=1024
        self.download_data_key = dict()
        self.download_data_key["wikitext"]= ("wikitext", "wikitext-2-raw-v1")
        self.download_data_key["shakespeare"]= ("tiny_shakespeare", None)
        self.download_data_key["ptb"]= ("ptb_text_only", None)
        self.download_data_key["news"]= ("ag_news", None)
        self.download_data_key["imdb"]= ("imdb", None)

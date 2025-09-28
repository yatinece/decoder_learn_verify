
from datasets import load_dataset
import argparse

# Creating dictionary of the dataset to use for experiments
download_data_key = dict()
download_data_key["wikitext"]= ("wikitext", "wikitext-2-raw-v1")
download_data_key["shakespeare"]= ("tiny_shakespeare", None)
download_data_key["ptb"]= ("ptb_text_only", None)
download_data_key["news"]= ("ag_news", None)
download_data_key["imdb"]= ("imdb", None)


#Using argument pareser to get the dataset

parser=argparse.ArgumentParser(description="Which dataset to download")

parser.add_argument("--dataset", type=str, choices=download_data_key.keys(), required=True)

args=parser.parse_args()

if  download_data_key[args.dataset][1] is not None :
    dataset = load_dataset(download_data_key[args.dataset][0], download_data_key[args.dataset][1])
else:
    dataset = load_dataset(download_data_key[args.dataset][0])

split_name = "train" if "train" in dataset else list(dataset.keys())[0]
print(f"Printing few rows from {args.dataset} ({split_name} split):")
print(dataset[split_name].select(range(5))["text"])


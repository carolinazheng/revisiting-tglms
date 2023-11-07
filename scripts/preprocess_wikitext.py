"""
Download and preprocess a WikiText dataset.
"""
import argparse
import os
import sys
from tqdm import tqdm
from datasets import load_dataset

sys.path.append("../")
from utils import NWL


def process_dataset(dataset):
    documents = []
    doc = []

    # Create list of documents, where each document is a list of tokens
    # newlines are replaced with special token <nwl>
    for i, line in enumerate(tqdm(dataset)):
        if len(line) == 0:
            line = "\n"

            # Check if we started a new section (next line is section header, and line after that is empty)
            if i < len(dataset) - 2 and dataset[i + 1][1:2] == "=" and len(dataset[i + 2]) == 0:
                # if so, append finished document to list
                if len(doc) > 0:
                    documents.append(doc)

                doc = []

        for token in line.split(" "):
            if token != "":
                if token == "\n":
                    token = NWL
                doc.append(token)

    if len(doc) > 0:
        documents.append(doc)

    # Convert each document from list of tokens to string
    for i, doc in enumerate(documents):
        documents[i] = " ".join(doc)

    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, default="wikitext-2")
    parser.add_argument("--n_docs", type=int, default=None)
    args = parser.parse_args()

    print("*" * 10)
    print(args)
    print("*" * 10)

    if args.n_docs is not None and args.n_docs <= 0:
        args.n_docs = None

    if not os.path.isdir(os.path.join(args.out_dir, args.config_name)):
        os.mkdir(os.path.join(args.out_dir, args.config_name))

    dataset = load_dataset("wikitext", args.config_name + "-v1")

    for split in ["train", "validation", "test"]:
        documents = process_dataset(dataset[split][: args.n_docs]["text"])
        out_split = "valid" if split == "validation" else split

        print(f"# {split} documents: {len(documents)}")

        with open(os.path.join(args.out_dir, args.config_name, f"{out_split}.txt"), "w") as f:
            f.write("\n".join(documents))
            f.write("\n")


if __name__ == "__main__":
    main()

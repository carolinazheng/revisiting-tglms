"""
Take the tokenized train data and drops all words that are not part of the TM vocab (run convert_vocab.py first to generate the vocab pickle). The resulting text files can be used in our coherence script.
"""
import argparse
import os
import pickle
from tqdm import tqdm


def process_texts(data_dir, split, tm_vocab):
    filtered_texts = []

    with open(os.path.join(data_dir, f"{split}.txt"), "r") as f:
        texts = [line.strip() for line in f.readlines()]

    for text in tqdm(texts):
        filtered_text = []

        for token in text.split():
            if token in tm_vocab:
                filtered_text.append(token)

        filtered_texts.append(filtered_text)

    return filtered_texts


def save_texts(out_dir, split, filtered_texts):
    with open(os.path.join(out_dir, f"{split}.txt"), "w") as f:
        for filtered_text in filtered_texts:
            f.write(" ".join(filtered_text) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="directory containing the tokenized train/valid/test files"
    )
    parser.add_argument("--vocab_path", type=str, required=True, help="path to save converted vocab pickle")
    parser.add_argument(
        "--out_dirname", type=str, default="coherence_texts", help="directory name to save coherence texts"
    )
    args = parser.parse_args()

    print("*" * 10)
    print(args)
    print("*" * 10)

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    out_dir = os.path.join(args.data_dir, args.out_dirname)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    tm_vocab = set(vocab["TM_vocab"])

    train_texts = process_texts(args.data_dir, "train", tm_vocab)
    valid_texts = process_texts(args.data_dir, "valid", tm_vocab)
    test_texts = process_texts(args.data_dir, "test", tm_vocab)

    save_texts(out_dir, "train", train_texts)
    save_texts(out_dir, "valid", valid_texts)
    save_texts(out_dir, "test", test_texts)


if __name__ == "__main__":
    main()

"""
Convert the vocab generated from our codebase's train script, saved as in the model checkpoint, to the vocab used by the rGBN-RNN codebase and our coherence script.
"""
import argparse
import os
import pickle
import torch
import numpy as np
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="directory containing model checkpoint")
    parser.add_argument("--out_path", type=str, required=True, help="path to save converted vocab pickle")
    args = parser.parse_args()

    print("*" * 10)
    print(args)
    print("*" * 10)

    checkpoint = torch.load(os.path.join(args.model_dir, "final_ckpt.pt"), map_location=device)
    vocab = checkpoint["vocab"]
    num_tm_words = checkpoint["model_cf"]["num_tm_words"]

    idxvocab = vocab.get_itos()
    vocabxid = defaultdict(int, vocab.get_stoi())
    ignore = set(range(num_tm_words, len(vocab)))
    TM_vocab = np.delete(idxvocab, list(ignore))

    with open(args.out_path, "wb") as f:
        pickle.dump(
            dict(
                idxvocab=idxvocab,
                vocabxid=vocabxid,
                ignore=ignore,
                TM_vocab=TM_vocab,
            ),
            f,
        )


if __name__ == "__main__":
    main()

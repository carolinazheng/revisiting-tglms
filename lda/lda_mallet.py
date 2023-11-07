"""
Train LDA using Mallet.
"""
import argparse
import numpy as np
import os
import random
import sys
import time
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers import LdaMallet

sys.path.append("../")
from data import get_data_iter, create_vocab, process_dataset
from utils import PAD, SOS, EOS, UNK, NWL, str_to_bool, get_topics_top_words


def main():
    parser = argparse.ArgumentParser(description="LDA train script")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset")
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--mallet_bin_path", type=str, required=True, help="path to mallet bin")
    parser.add_argument("--num_topics", type=int, default=100, help="number of topics")
    parser.add_argument("--num_iters", type=int, default=1000, help="number of Gibbs sampling iterations")
    parser.add_argument("--save_dir", type=str, default=None, help="path to save topics")
    parser.add_argument("--num_top_words", type=int, default=20, help="number of top words to save")
    parser.add_argument("--stopwords_file", type=str, default=None, help="path to stopwords file")
    parser.add_argument("--vocab_min_freq", type=int, default=10, help="minimum LM word frequency")
    parser.add_argument("--vocab_min_tm_freq", type=int, default=0, help="minimum TM word frequency")
    parser.add_argument("--vocab_min_tm_doc_freq", type=int, default=0, help="minimum TM document frequency")
    parser.add_argument(
        "--vocab_max_tm_freq_pct", type=float, default=0.001, help="fraction of top frequency words to exclude from TM"
    )
    parser.add_argument("--ignore_symbols", type=str_to_bool, default=True, help="ignore symbols in the TM")
    parser.add_argument(
        "--num_docs", type=int, default=None, help="max number of documents to use from dataset (for debugging)"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("*" * 10)
    print(args)
    print("*" * 10)

    print(f"Running LDA with {args.num_topics} topics for {args.num_iters} iterations.")

    if args.save_dir is None:
        print("Not saving!")
    else:
        print(f"Saving to {args.save_dir}.")

    if args.stopwords_file is not None:
        with open(args.stopwords_file, "r") as f:
            stopwords = set(f.read().splitlines())
    else:
        stopwords = {}

    def get_doc_bows(split="train"):
        with open(os.path.join(args.data_path, split + ".txt"), "r") as f:
            return f.readlines()

    docs = get_doc_bows(split="train")
    _, word_counter, word_doc_counter = get_data_iter(args.data_path, split="train")

    if args.dataset.startswith("wikitext"):
        dummy_symbols = [EOS, PAD, UNK, NWL]
    else:
        dummy_symbols = [SOS, EOS, PAD, UNK]

    vocab, num_tm_words = create_vocab(
        True,
        word_counter,
        word_doc_counter,
        stopwords,
        dummy_symbols,
        min_freq=args.vocab_min_freq,
        min_tm_freq=args.vocab_min_tm_freq,
        min_tm_doc_freq=args.vocab_min_tm_doc_freq,
        max_tm_freq_pct=args.vocab_max_tm_freq_pct,
        ignore_symbols=args.ignore_symbols,
    )

    print(f"Total vocab size: {len(vocab)}, # TM words: {num_tm_words}")
    print("Creating document-term matrix...")

    def create_X(docs):
        X = []

        for doc in docs:
            counts = defaultdict(int)

            for line in doc.split("\t"):
                for token in line.split(" "):
                    if vocab[token] < num_tm_words:
                        counts[vocab[token]] += 1

            X.append(sorted(counts.items()))

        return X

    if args.num_docs is not None:
        docs = docs[: args.num_docs]

    X = create_X(docs)
    itos = vocab.get_itos()[:num_tm_words]
    itos_dict = {}

    for i, s in enumerate(itos):
        itos_dict[i] = s

    id2word = Dictionary.from_corpus(X, id2word=itos_dict)

    print("Finished creating data, fitting LDA...")

    start = time.time()
    lda = LdaMallet(
        args.mallet_bin_path,
        corpus=X,
        num_topics=args.num_topics,
        id2word=id2word,
        iterations=args.num_iters,
        random_seed=args.seed,
    )
    print(f"Fitting took {time.time() - start:.2f} seconds.")

    if args.save_dir is None:
        get_topics_top_words(lda.get_topics(), vocab, topn=args.num_top_words, to_print=True)
    else:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        get_topics_top_words(
            lda.get_topics(),
            vocab,
            topn=args.num_top_words,
            to_print=True,
            fname=os.path.join(args.save_dir, f"topics_top{args.num_top_words}.txt"),
        )
        lda.save(os.path.join(args.save_dir, f"model.pkl"))


if __name__ == "__main__":
    main()

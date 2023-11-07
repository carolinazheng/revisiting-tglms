"""
Test script for TGLMs.
"""
import argparse
import os
import torch

import models
from train import evaluate
from data import PAD, process_dataset
from utils import str_to_bool, get_topics_top_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="TGLM test script")
    parser.add_argument("--load_dir", type=str, required=True, help="model checkpoint load directory")
    parser.add_argument(
        "--num_docs", type=int, default=None, help="max number of documents to use from dataset (for debugging)"
    )
    parser.add_argument("--data_path", type=str, default=None, help="path to dataset")
    parser.add_argument("--dataset", type=str, default=None, help="name of dataset")
    parser.add_argument("--max_seqlen", type=int, default=30, help="bptt/chunk sequence length")
    parser.add_argument("--sentence_level", type=str_to_bool, default=False, help="sentence-level LM")
    parser.add_argument(
        "--save_topics", type=str_to_bool, default=False, help="save topics to a file in checkpoint dir"
    )
    # TDLM-specific config
    parser.add_argument("--tm_seqlen", type=int, default=3, help="TM sequence length")
    parser.add_argument("--doc_len", type=int, default=300, help="TM document length as input to LM")
    parser.add_argument("--tm_batch_size", type=int, default=320, help="batch size for TM eval")

    args = parser.parse_args()
    print("*" * 10)
    print(args)
    print("*" * 10)

    print(f"Loading checkpoint from {args.load_dir}...")
    checkpoint = torch.load(os.path.join(args.load_dir, "final_ckpt.pt"), map_location=device)
    model = getattr(models, checkpoint["model_type"])(**checkpoint["model_cf"]).to(device)
    model.load_state_dict(checkpoint["model"])
    vocab = checkpoint["vocab"]
    num_tm_words = 0 if model.model_type == "LSTM_LM" else model.num_tm_words
    has_tm = model.model_type != "LSTM_LM" and (model.model_type != "TDLM" or not model.lm_only)

    if args.save_topics:
        if not has_tm:
            print(f"{model.model_type} does not have a TM, will not save topics.")
        else:
            print(f"Saving topics to {args.load_dir}.")
            get_topics_top_words(
                model.get_topics(), vocab, topn=20, fname=os.path.join(args.load_dir, "topics_top20.txt")
            )

    if args.dataset is not None:
        print(f"Loading test data...")
        _, _, test_data, _, _ = process_dataset(
            model.model_type,
            args.dataset,
            args.data_path,
            args.max_seqlen,
            testing=True,
            sentence_level=args.sentence_level,
            num_docs=args.num_docs,
            vocab=vocab,
            num_tm_words=model.num_tm_words,
            tm_sequence_length=args.tm_seqlen if model.model_type == "TDLM" else None,
            tm_doc_length=args.doc_len if model.model_type == "TDLM" else None,
        )

        test_data["lm_sequences"] = test_data["lm_sequences"].reshape((-1, 1))
        test_ppl, test_tm_ppl = evaluate(
            model, None, test_data, args.max_seqlen, vocab[PAD], None, tm_batch_size=args.tm_batch_size
        )

        if model.model_type == "TDLM":
            print(f"Evaluation | tst ppl {test_ppl:8.2f} | tst tm ppl {test_tm_ppl:8.2f}")
        else:
            print(f"Evaluation | tst ppl {test_ppl:8.2f}")


if __name__ == "__main__":
    main()

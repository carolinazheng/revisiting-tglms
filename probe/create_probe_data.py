"""
Compute the raw data used to train the probe.
This script will run the LSTM-LM or TGLM on the evaluation dataset and save the computed hidden states and targets.
"""
import argparse
import os
import random
import sys
import time
import torch
import numpy as np

sys.path.append("../")
import models
from data import PAD, process_dataset, create_lm_batched, get_lm_batch
from utils import seed_everything, str_to_bool, repackage_hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_name_map = {
    "LSTM_LM": "control_attention",
    "TDLM": "attention",
    "TRNN": "mu_theta",
}


def compute_hidden_states(model, batched_data, max_seqlen, out_dir, split):
    model.eval()
    start_time = time.time()
    batched_sequences = batched_data["lm_sequences"]
    batch_size = batched_sequences.shape[-1]
    hidden = model.init_hidden(batch_size)
    num_batches = len(batched_sequences) // max_seqlen
    is_last = [1] * batch_size
    data_prev = None

    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch = get_lm_batch(batched_data, batch_idx, max_seqlen, device)
            hidden = repackage_hidden(hidden, is_last)
            data, is_last = batch["data"], batch["is_last"].tolist()

            if model.model_type == "LSTM_LM":
                output, hidden_new = model(data, hidden)
                probe_target = None
            elif model.model_type == "TDLM":
                attention, hidden_new = model(data, hidden, batch["cum_tm_docs"], "lm", return_attention=True)
                probe_target = attention
            elif model.model_type == "TRNN":
                output, hidden_new = model(data, batch["cum_bows"], hidden)
                probe_target = output["mu_theta"]

            torch.save(
                {"hidden": hidden, "is_last": is_last, target_name_map[model.model_type]: probe_target},
                os.path.join(out_dir, f"{split}_{batch_idx}.pt"),
            )
            hidden = hidden_new
            data_prev = data

            print(f"{batch_idx + 1}/{num_batches} {split} batches", end="\r" if batch_idx < num_batches - 1 else "\n")

    print(f"Finished {split} in {round(time.time() - start_time, 2)}s.")


def main():
    parser = argparse.ArgumentParser(description="Create (hidden, target) data for probe")
    parser.add_argument("--model_dir", type=str, required=True, help="directory containing model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save created probe data")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--max_seqlen", type=int, default=30, help="bptt/chunk sequence length")
    parser.add_argument("--tm_seqlen", type=int, default=3, help="TM sequence length")
    parser.add_argument("--doc_len", type=int, default=300, help="TM document length as input to LM")
    parser.add_argument(
        "--num_docs", type=int, default=None, help="max number of documents to use from dataset (for debugging)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    print("*" * 10)
    print(args)
    print("*" * 10)

    checkpoint = torch.load(os.path.join(args.model_dir, "final_ckpt.pt"))
    model_cf = checkpoint["model_cf"]
    model_type = checkpoint["model_type"]
    model = getattr(models, model_type)(**model_cf).to(device)
    model.load_state_dict(checkpoint["model"])
    vocab = checkpoint["vocab"]
    num_tm_words = 0 if model_type == "LSTM_LM" else model.num_tm_words

    train_data, val_data, test_data, _, _ = process_dataset(
        model_type,
        args.dataset,
        args.data_path,
        args.max_seqlen,
        vocab=vocab,
        num_tm_words=num_tm_words,
        tm_sequence_length=args.tm_seqlen,
        tm_doc_length=args.doc_len,
        num_docs=args.num_docs,
    )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for split, data in zip(["train", "valid", "test"], [train_data, val_data, test_data]):
        if model_type == "LSTM_LM":
            batched_data = create_lm_batched(data, args.batch_size)
            compute_hidden_states(model, batched_data, args.max_seqlen, args.out_dir, split)
        elif model_type == "TDLM":
            batched_data = create_lm_batched(
                data, args.batch_size, ignore_keys=["all_tm_docs", "tm_sequences", "tm_docs"]
            )
            compute_hidden_states(model, batched_data, args.max_seqlen, args.out_dir, split)
        elif model_type == "TRNN":
            batched_data = create_lm_batched(data, args.batch_size, ignore_keys=["all_bows"])
            compute_hidden_states(model, batched_data, args.max_seqlen, args.out_dir, split)


if __name__ == "__main__":
    main()

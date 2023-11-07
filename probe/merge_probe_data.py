"""
Create the actual training data for the probe.
This script merges the data (hidden states from the LSTM-LM) and targets (attention from TDLM or mu_theta from TRNN).
"""
import argparse
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_split(split, target_name, data_dir_tglm, data_dir_lstm_lm, out_dir):
    # Get batch size
    with open(os.path.join(data_dir_tglm, f"{split}_0.pt"), "rb") as f:
        batch_size = len(torch.load(f, map_location=device)["is_last"])

    # Get number of batches
    num_batches = 0
    path = os.path.join(data_dir_tglm, f"{split}_{num_batches}.pt")

    while os.path.exists(path):
        num_batches += 1
        path = os.path.join(data_dir_tglm, f"{split}_{num_batches}.pt")

    # Helper functions
    def flatten_lstm_hidden(hidden):
        return torch.cat((hidden[0].squeeze(), hidden[1].squeeze()), dim=1)

    def init_cur_batch():
        batch = {}
        for key in ["hidden_tglm", "hidden_lstm_lm", "target"]:
            batch[key] = torch.tensor([], device=device)
        return batch

    def append_to_batch(cur_batch, batch, indices):
        cur_batch["hidden_tglm"] = torch.cat((cur_batch["hidden_tglm"], batch["hidden_tglm"][indices]), dim=0)
        cur_batch["hidden_lstm_lm"] = torch.cat((cur_batch["hidden_lstm_lm"], batch["hidden_lstm_lm"][indices]), dim=0)
        cur_batch["target"] = torch.cat((cur_batch["target"], batch[target_name][indices]), dim=0)

    is_last_prev = np.ones(batch_size)
    cur_batch = init_cur_batch()
    cur_batch_idx = 0

    for batch_idx in range(num_batches):
        with open(os.path.join(data_dir_tglm, f"{split}_{batch_idx}.pt"), "rb") as f:
            tglm_batch = torch.load(f, map_location=device)

        with open(os.path.join(data_dir_lstm_lm, f"{split}_{batch_idx}.pt"), "rb") as f:
            lstm_lm_batch = torch.load(f, map_location=device)

        # hiddens are now bsz x (hidden_dim * 2)
        tglm_batch["hidden_tglm"] = flatten_lstm_hidden(tglm_batch["hidden"])
        tglm_batch["hidden_lstm_lm"] = flatten_lstm_hidden(lstm_lm_batch["hidden"])

        # Find indices where the previous is_last was 0 (i.e., didn't reset hidden state)
        indices = np.nonzero(1 - is_last_prev)[0]
        num_indices = len(indices)
        num_cur_batch = len(cur_batch["hidden_tglm"])

        # If we have enough new examples for a new batch
        if num_indices + num_cur_batch >= batch_size:
            num_cur_indices = batch_size - num_cur_batch
            append_to_batch(cur_batch, tglm_batch, indices[:num_cur_indices])

            torch.save(cur_batch, os.path.join(out_dir, f"{split}_{cur_batch_idx}.pt"))
            cur_batch_idx += 1
            del cur_batch
            cur_batch = init_cur_batch()

            if num_cur_indices < num_indices:
                append_to_batch(cur_batch, tglm_batch, indices[num_cur_indices:])
        else:
            append_to_batch(cur_batch, tglm_batch, indices)

        is_last_prev = np.array(tglm_batch["is_last"])
        del tglm_batch, lstm_lm_batch

        if cur_batch_idx % 100 == 0 or cur_batch_idx == num_batches - 1:
            print(f"{batch_idx}/{num_batches} processed...", end="\r" if cur_batch_idx < num_batches - 1 else "\n")

    print(f"# merged {split} batches: {cur_batch_idx}")


def main():
    parser = argparse.ArgumentParser(description="Merge data from LSTM-LM and TGLM into (hidden, target) pairs")

    parser.add_argument("--data_dir_tglm", type=str, required=True, help="path to TGLM data folder")
    parser.add_argument("--data_dir_lstm_lm", type=str, required=True, help="path to LSTM-LM data folder")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument(
        "--target_name", type=str, required=True, help="name of target (attention, control_attention, or mu_theta)"
    )
    parser.add_argument("--split", type=str, default=None, help="train, valid, or test")

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    splits = ["train", "valid", "test"] if args.split is None else [args.split]
    print(f"Splits: {splits}")

    for split in splits:
        process_split(split, args.target_name, args.data_dir_tglm, args.data_dir_lstm_lm, args.out_dir)


if __name__ == "__main__":
    main()

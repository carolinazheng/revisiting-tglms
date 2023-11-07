"""
Train script for the probe.
"""
import argparse
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
import wandb
from copy import deepcopy
from datetime import datetime
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("../")
from utils import str_to_bool, log, log_dict, seed_everything


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = torch.finfo(torch.float32).eps


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.linear(input)


class ProbeDataset(Dataset):
    def __init__(self, data_dir, split, inverse_softmax=False, num_docs=None):
        self.split = split
        self.data_dir = data_dir
        self.inverse_softmax = inverse_softmax
        # Compute number of batches
        self.num_batches = 0
        path = os.path.join(data_dir, f"{split}_{self.num_batches}.pt")

        while os.path.exists(path):
            if num_docs is not None and self.num_batches >= num_docs:
                break

            self.num_batches += 1
            path = os.path.join(data_dir, f"{split}_{self.num_batches}.pt")

        print(f"Prepared {split} data: {self.num_batches} batches")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, f"{self.split}_{idx}.pt")

        with open(path, "rb") as f:
            batch = torch.load(f, map_location=device)

        if self.inverse_softmax:
            batch["target"] = inverse_softmax(batch["target"])

        return batch


def inverse_softmax(target):
    target = (target + epsilon).log()
    return target - target.mean(dim=1).unsqueeze(1)


def compute_sample_mean(loader):
    tot = 0.0

    for sample in loader:
        tot += torch.mean(sample["target"].squeeze(), dim=0)

    return (tot / len(loader)).to(device)


def evaluate(model, loader, sample_mean, epoch=None, writer=None):
    model.eval()
    criterion = nn.MSELoss()
    total_accuracy = 0.0
    total_accuracy_top5 = 0.0
    ss_tot = 0.0
    ss_res = 0.0
    total_mse = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = sample["hidden_lstm_lm"].squeeze()
            target = sample["target"].squeeze()
            acc_target = torch.max(target, dim=1)[1]
            output = model(data)

            accuracy = (acc_target == torch.max(output, dim=1)[1]).sum() / len(acc_target)
            total_accuracy += accuracy.item()
            total_mse += criterion(output, target)
            ss_tot += torch.sum((target - sample_mean) ** 2)
            ss_res += torch.sum((target - output) ** 2)
            accuracy_top5 = 0.0
            sorted_output = torch.sort(output, descending=True)[1][:, :5]

            for i, row in enumerate(sorted_output):
                accuracy_top5 += acc_target[i] in row

            total_accuracy_top5 += accuracy_top5 / len(acc_target)

    metrics = {
        "accuracy": total_accuracy / num_batches,
        "accuracy-top5": total_accuracy_top5 / num_batches,
        "mse": total_mse / num_batches,
        "r2": 1 - ss_res / ss_tot,
    }

    split = "test" if epoch is None else "valid"
    log_dict(writer, split, metrics, epoch)

    return metrics


def train(model, loader, optimizer, epoch, writer=None):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(loader)

    for sample in tqdm(loader):
        input = sample["hidden_lstm_lm"].squeeze()
        target = sample["target"].squeeze()
        acc_target = torch.max(target, dim=1)[1]
        output = model(input)
        optimizer.zero_grad()
        loss = criterion(output, target)
        accuracy = (acc_target == torch.max(output, dim=1)[1]).sum() / len(acc_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy.item()

    log(writer, "train", "mse", total_loss / num_batches, epoch)

    return total_loss / num_batches, total_accuracy / num_batches


def make_config(args, input_dim, output_dim):
    model_cf = {
        "input_dim": input_dim,
        "output_dim": output_dim,
    }

    writer_cf = dict(model_cf)
    writer_params = ["model_type", "inverse_softmax", "num_epochs", "lr", "wdecay", "patience", "seed"]
    writer_cf.update({param: getattr(args, param) for param in writer_params})

    return model_cf, writer_cf


def main():
    parser = argparse.ArgumentParser(description="Train probe")
    parser.add_argument("--data_dir", type=str, required=True, help="path to *merged* data folder")
    parser.add_argument("--save_dir", type=str, default=None, help="model save dir")
    parser.add_argument(
        "--inverse_softmax", type=str_to_bool, default=False, help="whether to transform target with inverse softmax"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--wdecay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--patience", type=int, default=None, help="number of epochs to wait for val r2 to improve")
    parser.add_argument("--writer_type", type=str, default=None, help="wandb or tensorboard")
    parser.add_argument("--writer_dir", type=str, default=None, help="tensorboard writer directory")
    parser.add_argument("--run_name", type=str, default=None, help="writer run name")
    parser.add_argument(
        "--num_docs", type=int, default=None, help="max number of documents to use from dataset (for debugging)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    print("*" * 10)
    print(args)
    print("*" * 10)

    seed_everything(args.seed)

    train_data = ProbeDataset(args.data_dir, "train", inverse_softmax=args.inverse_softmax, num_docs=args.num_docs)
    val_data = ProbeDataset(args.data_dir, "valid", inverse_softmax=args.inverse_softmax, num_docs=args.num_docs)
    test_data = ProbeDataset(args.data_dir, "test", inverse_softmax=args.inverse_softmax, num_docs=args.num_docs)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    input_dim = train_data[0]["hidden_lstm_lm"].size(1)  # hidden dim
    output_dim = train_data[0]["target"].size(1)  # number of TDLM topics
    print(f"Probe input dim: {input_dim}, probe output dim: {output_dim}")

    model_cf, writer_cf = make_config(args, input_dim, output_dim)

    if args.writer_type == "wandb":
        run = wandb.init(project="[project]", entity="[entity]", config=writer_cf, name=args.run_name)
        run.define_metric("train/mse", step_metric="epoch")
        run.define_metric("valid/mse", step_metric="epoch")
        run.define_metric("valid/r2", step_metric="epoch")
        run.define_metric("valid/accuracy", step_metric="epoch")
        writer = wandb
    elif args.writer_type == "tensorboard":
        writer_dir = "runs" if args.writer_dir is None else args.writer_dir
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{os.path.dirname(args.data_dir)}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        writer = SummaryWriter(log_dir=f"{writer_dir}/{run_name}")
    else:
        print("Not logging to wandb or tensorboard.")
        writer = None

    model = LinearModel(**model_cf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    sample_mean_val = compute_sample_mean(val_loader)

    best_epoch = 0
    best_val_r2 = float("-inf")
    best_model_state = model.state_dict()
    best_optim_state = None
    patience_count = 0

    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, epoch, writer=writer)
        val_metrics = evaluate(model, val_loader, sample_mean_val, epoch=epoch, writer=writer)

        print(
            f"Epoch {epoch}: train mse {train_loss:.5f} | val mse {val_metrics['mse']:.5f} | val r2 {val_metrics['r2']:.5f} | val acc {val_metrics['accuracy']:.5f}"
        )

        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            best_optim_state = deepcopy(optimizer.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if args.patience is not None and patience_count >= args.patience:
                print(f"Val R2 hasn't improved for {args.patience} epochs, stopping.")
                break

    model.load_state_dict(best_model_state)
    sample_mean_test = compute_sample_mean(test_loader)
    test_metrics = evaluate(model, test_loader, sample_mean_test, writer=writer)
    print("=" * 20)
    print(f"Training completed. Epoch with best val R2: {best_epoch}")
    print(
        f"test mse {test_metrics['mse']:.5f} | test r2 {test_metrics['r2']:.5f} | test top-1 acc {test_metrics['accuracy']:.5f} | test top-5 acc {test_metrics['accuracy-top5']:.5f}"
    )

    if args.writer_type == "tensorboard":
        writer.add_hparams(writer_cf, {"valid/mse": best_val_r2})

    if args.save_dir is not None:
        print(f"Saving model to {args.save_dir}")

        checkpoint = {
            "epoch": args.num_epochs,
            "best_epoch": best_epoch,
            "model_type": args.model_type,
            "model_cf": model_cf,
            "model": best_model_state,
            "optimizer": best_optim_state,
            "seed": args.seed,
            "wandb_id": wandb.run.id if args.writer_type == "wandb" else None,
        }

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        torch.save(checkpoint, os.path.join(args.save_dir, "final_ckpt.pt"))


if __name__ == "__main__":
    main()

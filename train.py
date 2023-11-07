"""
Train script for TGLMs.
"""
import argparse
import numpy as np
import os
import time
import torch
import wandb
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime
from torch.distributions import Dirichlet
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

import models
from data import process_dataset, create_lm_batched, get_lm_batch, get_tm_batch
from utils import (
    PAD,
    supported_datasets,
    supported_model_types,
    supported_optimizers,
    log_metric_names,
    str_to_bool,
    seed_everything,
    set_other_model_type_args_to_none,
    make_config,
    repackage_hidden,
    init_embeddings,
    tdlm_shuffle_lm_tm_batch_idxs,
    get_topics_top_words,
    log,
    log_dict,
    log_topics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = torch.finfo(torch.float32).eps


def compute_loss_ll(output, target, pad_idx):
    criterion = torch.nn.NLLLoss(ignore_index=pad_idx)
    loss = criterion(output, target)
    num_tokens = (target != pad_idx).sum().item()

    return dict(total_loss=loss, num_tokens=num_tokens)


def compute_loss_elbo_vrtm(output, target, pad_idx, doc_lens, kl_weight=1.0):
    mask = (target != pad_idx).float()
    num_tokens = mask.sum()

    # L_w
    token_criterion = torch.nn.NLLLoss(ignore_index=pad_idx)
    phi = output["q_phis"]
    beta = output["topics"]
    num_topics, num_tm_words = beta.shape
    rnn_logits_all = output["rnn_logits"]
    vocab_size = rnn_logits_all.shape[-1]
    rnn_logits = rnn_logits_all[:, :, :num_tm_words]
    tm_logits = phi @ beta
    target_stopwords = (target >= num_tm_words).view(rnn_logits.shape[0], rnn_logits.shape[1]).float()

    log_Z = torch.logsumexp(rnn_logits.unsqueeze(2) + beta.unsqueeze(0).unsqueeze(0), dim=-1)
    log_Z_logits = torch.einsum("sbi,sbi->sb", log_Z, phi)
    mixed_logits = rnn_logits + tm_logits + log_Z_logits.unsqueeze(-1)
    mixed_logits_all = torch.full_like(rnn_logits_all, float("-inf"))
    mixed_logits_all[:, :, :num_tm_words] = mixed_logits

    logits_all = torch.where(target_stopwords.unsqueeze(-1).bool(), rnn_logits_all, mixed_logits_all)
    token_loss = token_criterion(F.log_softmax(logits_all, dim=-1).reshape(-1, vocab_size), target)

    # L_l
    stopword_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    stopword_loss = (
        mask * stopword_criterion(output["stopword_logits"], target_stopwords.view(-1))
    ).sum() / num_tokens

    # L_z + L_phi
    theta = output["q_thetas_sampled"]
    kl_loss_z = (
        -(
            torch.einsum("sbk,sbk->sb", phi, ((theta + epsilon).log() - (phi + epsilon).log()))
            * (1 - target_stopwords)
        ).sum()
        / num_tokens
    )

    # L_theta
    gamma = output["q_gammas"] + epsilon
    alpha = torch.full_like(gamma, 1 / num_topics)
    kl_loss_theta = (
        kl_divergence(Dirichlet(gamma), Dirichlet(torch.full_like(gamma, 1 / num_topics))) / doc_lens
    ).mean()

    kl_loss = kl_loss_theta + kl_loss_z

    return {
        "token_loss": token_loss.item(),
        "stopword_loss": stopword_loss.item(),
        "kl_loss_theta": kl_loss_theta.item(),
        "kl_loss_z": kl_loss_z.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": token_loss + stopword_loss + kl_weight * kl_loss,
        "num_tokens": num_tokens.item(),
    }


def compute_loss_elbo_trnn(output, target, pad_idx, doc_lens, num_tm_words, kl_weight=1.0):
    token_criterion = torch.nn.NLLLoss(ignore_index=pad_idx)
    stopword_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    batch_size = len(doc_lens)
    mask = (target != pad_idx).float()
    num_tokens = mask.sum()
    target_stopwords = (target >= num_tm_words).float()
    token_logits = torch.where(
        target_stopwords.unsqueeze(1).bool(), output["token_rnn_output"], output["token_mixed_output"]
    )
    mu, logsigma = output["mu_theta"], output["logsigma_theta"]

    kl_loss = (0.5 * torch.sum(mu.pow(2) + logsigma.exp() - logsigma - 1, dim=-1) / doc_lens).sum() / batch_size
    token_loss = token_criterion(token_logits, target)
    stopword_loss = (mask * stopword_criterion(output["stopword_output"], target_stopwords)).sum() / num_tokens

    return {
        "token_loss": token_loss.item(),
        "stopword_loss": stopword_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": token_loss + stopword_loss + kl_weight * kl_loss,
        "num_tokens": num_tokens.item(),
    }


def compute_log_ppl(model_type, output, target, pad_idx, prev_thetas=None):
    if model_type == "LSTM_LM" or model_type == "TDLM":
        criterion = torch.nn.NLLLoss(ignore_index=pad_idx)

        with torch.no_grad():
            log_ppl = criterion(output, target).item()

        return log_ppl
    elif model_type == "TRNN":
        return compute_log_ppl_trnn(output, target, pad_idx)
    elif model_type == "VRTM":
        return compute_log_ppl_vrtm(output, target, pad_idx, prev_thetas=prev_thetas)


def compute_log_ppl_trnn(output, target, pad_idx):
    criterion = torch.nn.NLLLoss(ignore_index=pad_idx)

    with torch.no_grad():
        stopword_prob = torch.sigmoid(output["stopword_output"]).unsqueeze(-1)
        output = (
            stopword_prob * output["token_rnn_output"].exp() + (1 - stopword_prob) * output["token_mixed_output"].exp()
        ).log()

    return criterion(output, target).item()


def compute_log_ppl_vrtm(output, target, pad_idx, prev_thetas=None):
    criterion = torch.nn.NLLLoss(ignore_index=pad_idx)

    with torch.no_grad():
        beta = output["topics"]
        theta = output["q_thetas_sampled"] if prev_thetas is None else prev_thetas
        rnn_logits_all = output["rnn_logits"]

        _, num_tm_words = beta.shape
        vocab_size = rnn_logits_all.size(-1)

        stopword_prob = torch.sigmoid(output["stopword_logits"]).unsqueeze(1)
        word_prob_stop = torch.softmax(rnn_logits_all, dim=-1).view(-1, vocab_size)

        rnn_logits = output["rnn_logits"][:, :, :num_tm_words]
        mixed_logits = rnn_logits.unsqueeze(2) + beta.unsqueeze(0).unsqueeze(0)
        mixed_logits = torch.einsum("sbkv,bk->sbv", mixed_logits, theta)
        mixed_logits_all = torch.full_like(rnn_logits_all, float("-inf"))
        mixed_logits_all[:, :, :num_tm_words] = mixed_logits
        word_prob_topic = torch.softmax(mixed_logits_all, dim=-1).view(-1, vocab_size)

        word_prob = stopword_prob * word_prob_stop + (1 - stopword_prob) * word_prob_topic

    return criterion(word_prob.log(), target).item()


def evaluate(model, epoch, unbatched_data, max_seqlen, pad_idx, writer, tm_batch_size=None):
    model.eval()
    total_loss = 0.0
    total_num_tokens = 0
    total_log_ppl = 0.0
    hidden = model.init_hidden(1)
    num_lm_batches = len(unbatched_data["lm_sequences"]) // max_seqlen
    is_last = 1
    prev_thetas = None

    if model.model_type == "TDLM" and not model.lm_only:
        total_tm_loss = 0.0
        total_tm_num_tokens = 0
        num_tm_batches = len(unbatched_data["tm_sequences"]) // tm_batch_size
        tm_sequence_length = len(unbatched_data["tm_sequences"][0][0])
    else:
        num_tm_batches = 0

    with torch.no_grad():
        for batch_idx in range(num_lm_batches):
            batch = get_lm_batch(unbatched_data, batch_idx, max_seqlen, device)
            hidden = repackage_hidden(hidden, is_last)
            data, target, is_last = batch["data"], batch["target"], batch["is_last"].tolist()

            if batch_idx == num_lm_batches - 1:
                data = data[:-1]
            elif is_last:
                target[-1] = pad_idx

            if model.model_type == "LSTM_LM":
                output, hidden = model(data, hidden)
                loss_dict = compute_loss_ll(output, target, pad_idx)
            elif model.model_type == "TRNN":
                output, hidden = model(data, batch["cum_bows"], hidden)
                loss_dict = compute_loss_elbo_trnn(output, target, pad_idx, batch["doc_lens"], model.num_tm_words)
            elif model.model_type == "VRTM":
                output, hidden = model(data, target, hidden)
                loss_dict = compute_loss_elbo_vrtm(output, target, pad_idx, batch["doc_lens"])
            elif model.model_type == "TDLM":
                batch_tm_docs = (
                    torch.full_like(batch["cum_tm_docs"], pad_idx) if model.lm_only else batch["cum_tm_docs"]
                )
                output, hidden = model(data, hidden, batch_tm_docs, "lm")
                loss_dict = compute_loss_ll(output, target, pad_idx)

            num_tokens = loss_dict["num_tokens"]

            if num_tokens > 0:
                total_loss += loss_dict["total_loss"].item() * num_tokens
                total_num_tokens += num_tokens
                total_log_ppl += (
                    compute_log_ppl(model.model_type, output, target, pad_idx, prev_thetas=prev_thetas) * num_tokens
                )

            if model.model_type == "VRTM":
                prev_thetas = output["q_thetas_sampled"] if is_last == 0 else None

            del batch

        for batch_idx in range(num_tm_batches):
            batch = get_tm_batch(unbatched_data, batch_idx, tm_batch_size, device)
            data, target = batch["data"], batch["target"]

            output = model(None, None, data, "tm")
            output = torch.repeat_interleave(output, tm_sequence_length, dim=0)
            loss_dict = compute_loss_ll(output, target, pad_idx)
            loss = loss_dict["total_loss"]
            num_tm_tokens = loss_dict["num_tokens"]
            total_tm_loss += loss.item() * num_tm_tokens
            total_tm_num_tokens += num_tm_tokens

            del data, target

    loss = total_loss / total_num_tokens
    ppl = np.exp(total_log_ppl / total_num_tokens)
    split = "test" if epoch is None else "valid"
    log(writer, split, "ppl", ppl, epoch)

    if model.model_type == "TDLM" and not model.lm_only:
        tm_ppl = np.exp(total_tm_loss / total_tm_num_tokens)
        log(writer, split, "tm_ppl", tm_ppl, epoch)
    else:
        tm_ppl = None

    return ppl, tm_ppl


def train(
    model,
    epoch,
    batched_data,
    max_seqlen,
    pad_idx,
    lm_optimizer,
    tm_optimizer,
    writer,
    kl_weight=1.0,
    tm_batch_size=None,
):
    log_interval = 200
    model.train()
    total_loss = 0.0
    total_num_tokens = 0
    interval_total_loss = 0.0
    interval_num_tokens = 0

    total_log_ppl = 0.0
    total_token_loss = 0.0
    total_stopword_loss = 0.0
    total_kl_loss = 0.0

    start_time = time.time()
    batch_size = batched_data["lm_sequences"].shape[-1]
    num_lm_batches = len(batched_data["lm_sequences"]) // max_seqlen
    hidden = model.init_hidden(batch_size)
    is_last = [1] * batch_size

    if model.model_type == "TDLM" and not model.lm_only:
        total_tm_loss = 0.0
        total_tm_num_tokens = 0
        tm_sequence_length = len(batched_data["tm_sequences"][0][0])
        num_tm_batches = len(batched_data["tm_sequences"]) // tm_batch_size
        batch_idxs = tdlm_shuffle_lm_tm_batch_idxs(num_lm_batches, num_tm_batches)
    else:
        batch_idxs = [("lm", i) for i in range(num_lm_batches)]

    for mode, batch_idx in batch_idxs:
        if mode == "lm":
            batch = get_lm_batch(batched_data, batch_idx, max_seqlen, device)
            # if the prev sequence was last in a document, reset the hidden state of the next sequence
            hidden = repackage_hidden(hidden, is_last)
            data, target, is_last = batch["data"], batch["target"], batch["is_last"].tolist()

            lm_optimizer.zero_grad()
            if tm_optimizer is not None:
                tm_optimizer.zero_grad()

            if model.model_type == "LSTM_LM":
                output, hidden = model(data, hidden)
                loss_dict = compute_loss_ll(output, target, pad_idx)
            elif model.model_type == "TRNN":
                output, hidden = model(data, batch["all_bows"], hidden)
                loss_dict = compute_loss_elbo_trnn(
                    output, target, pad_idx, batch["doc_lens"], model.num_tm_words, kl_weight=kl_weight
                )
            elif model.model_type == "VRTM":
                output, hidden = model(data, target, hidden)
                loss_dict = compute_loss_elbo_vrtm(output, target, pad_idx, batch["doc_lens"], kl_weight=kl_weight)
            elif model.model_type == "TDLM":
                batch_tm_docs = (
                    torch.full_like(batch["all_tm_docs"], pad_idx) if model.lm_only else batch["all_tm_docs"]
                )
                output, hidden = model(data, hidden, batch["all_tm_docs"], "lm")
                loss_dict = compute_loss_ll(output, target, pad_idx)

            loss = loss_dict["total_loss"]
            num_tokens = loss_dict["num_tokens"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            lm_optimizer.step()
            if tm_optimizer is not None:
                tm_optimizer.step()
            log_ppl = compute_log_ppl(model.model_type, output, target, pad_idx)
            del batch

            total_loss += loss.item() * num_tokens
            total_num_tokens += num_tokens
            interval_total_loss += loss.item() * num_tokens
            interval_num_tokens += num_tokens

            total_log_ppl += log_ppl * num_tokens

            if model.model_type == "VRTM" or model.model_type == "TRNN":
                total_token_loss += loss_dict["token_loss"] * num_tokens
                total_stopword_loss += loss_dict["stopword_loss"] * num_tokens
                total_kl_loss += loss_dict["kl_loss"] * num_tokens

            if batch_idx % log_interval == 0:
                cur_loss = interval_total_loss / interval_num_tokens
                elapsed = time.time() - start_time

                print(
                    f"| epoch {epoch:3d} | {batch_idx:5d}/{num_lm_batches} batches | ms/batch {elapsed * 1000 / log_interval:5.2f} | loss {cur_loss:5.2f}",
                    end="\r" if batch_idx < num_lm_batches - 1 else "\n",
                )

                interval_total_loss = 0.0
                interval_num_tokens = 0
                start_time = time.time()
        else:
            # We are training TDLM's topic model
            batch = get_tm_batch(batched_data, batch_idx, tm_batch_size, device)
            data, target = batch["data"], batch["target"]
            tm_optimizer.zero_grad()
            output = model(None, None, data, "tm")
            output = torch.repeat_interleave(output, tm_sequence_length, dim=0)
            loss_dict = compute_loss_ll(output, target, pad_idx)
            loss = loss_dict["total_loss"]
            num_tm_tokens = loss_dict["num_tokens"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            tm_optimizer.step()
            del batch

            total_tm_loss += loss.item() * num_tm_tokens
            total_tm_num_tokens += num_tm_tokens

    train_metrics = dict(
        loss=total_loss / total_num_tokens,
        ppl=np.exp(total_log_ppl / total_num_tokens),
    )

    if model.model_type == "VRTM" or model.model_type == "TRNN":
        train_metrics.update(
            dict(
                token_loss=total_token_loss / total_num_tokens,
                stopword_loss=total_stopword_loss / total_num_tokens,
                kl_loss=total_kl_loss / total_num_tokens,
                kl_weight=kl_weight,
            )
        )
    elif model.model_type == "TDLM" and not model.lm_only:
        train_metrics["tm_ppl"] = np.exp(total_tm_loss / total_tm_num_tokens)

    log_dict(writer, "train", train_metrics, epoch)


def main():
    parser = argparse.ArgumentParser(description="TGLM train script")
    # Log/save/debug
    parser.add_argument(
        "--num_docs", type=int, default=None, help="max number of documents to use from dataset (for debugging)"
    )
    parser.add_argument("--save_dir", type=str, help="model save directory")
    parser.add_argument("--writer_type", type=str, default=None, help="tensorboard, wandb, or none")
    parser.add_argument("--writer_dir", type=str, default=None, help="tensorboard writer save directory")
    parser.add_argument("--run_name", type=str, default=None, help="writer run name")
    # Data/vocab
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset")
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--pretrained_emb_file", type=str, help="path to pretrained embeddings")
    parser.add_argument("--stopwords_file", type=str, default=None, help="path to stopwords file")
    parser.add_argument("--vocab_min_freq", type=int, default=10, help="minimum LM word frequency")
    parser.add_argument("--vocab_min_tm_freq", type=int, default=0, help="minimum TM word frequency")
    parser.add_argument("--vocab_min_tm_doc_freq", type=int, default=0, help="minimum TM document frequency")
    parser.add_argument(
        "--vocab_max_tm_freq_pct", type=float, default=0.001, help="fraction of top frequency words to exclude from TM"
    )
    parser.add_argument("--ignore_symbols", type=str_to_bool, default=True, help="ignore symbols in the TM")
    # Model config
    parser.add_argument("--model_type", type=str, required=True, help="LSTM_LM, TRNN, VRTM, or TDLM")
    parser.add_argument("--embedding_dim", type=int, default=300, help="input embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=600, help="LM hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout")
    parser.add_argument("--num_layers", type=int, default=1, help="LM number of layers")
    parser.add_argument("--num_topics", type=int, default=50, help="number of topics")
    # VRTM/TRNN
    parser.add_argument("--hidden_dim_theta", type=int, default=256, help="theta encoder hidden layer dimension")
    parser.add_argument(
        "--warmup_kl", type=int, default=0, help="# epochs to linearly increase KL weight from kl_start"
    )
    parser.add_argument("--kl_start", type=float, default=1.0, help="starting KL weight")
    parser.add_argument(
        "--use_vrtm_stopword_net", type=str_to_bool, default=True, help="use stopword net from VRTM paper"
    )
    # TDLM
    parser.add_argument("--tm_batch_size", type=int, default=320, help="batch size for TM training")
    parser.add_argument(
        "--lm_only", type=str_to_bool, default=False, help="use TDLM architecture, but replace documents with padding"
    )
    parser.add_argument("--filter_sizes", type=list, default=[2], help="TM encoder filter sizes")
    parser.add_argument("--num_filters", type=int, default=20, help="TM encoder number of filters")
    parser.add_argument("--topic_embedding_size", type=int, default=50, help="TM topic embedding size")
    parser.add_argument("--tm_keep_prob", type=float, default=0.4, help="TM dropout")
    parser.add_argument("--lm_keep_prob", type=float, default=0.6, help="LM dropout")
    parser.add_argument("--tm_seqlen", type=int, default=3, help="TM sequence length")
    parser.add_argument("--doc_len", type=int, default=300, help="TM document length as input to LM")
    # Train settings
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--lm_lr", type=float, default=0.001, help="LM learning rate")
    parser.add_argument("--tm_lr", type=float, default=0.001, help="TM learning rate")
    parser.add_argument("--lm_optimizer", type=str, default="adam", help="LM optimizer (sgd or adam)")
    parser.add_argument("--wdecay", type=float, default=0, help="weight decay")
    parser.add_argument("--patience", type=int, default=0, help="patience for early stopping")
    parser.add_argument("--stopping_threshold", type=float, default=1e-3, help="early stopping threshold")
    parser.add_argument("--max_seqlen", type=int, default=30, help="bptt/chunk sequence length")
    parser.add_argument("--sentence_level", type=str_to_bool, default=False, help="sentence-level LM")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    if args.dataset not in supported_datasets:
        raise Exception(f"Unsupported dataset {args.dataset}.")
    if args.model_type not in supported_model_types:
        raise Exception(f"Unsupported model type {args.model_type}.")
    if args.lm_optimizer not in supported_optimizers:
        raise Exception(f"Unsupported lm optimizer {args.lm_optimizer}.")

    set_other_model_type_args_to_none(args)
    seed_everything(args.seed)
    print("*" * 10)
    print(args)
    print("*" * 10)

    print(f"Loading data...")
    train_data, val_data, test_data, vocab, num_tm_words = process_dataset(
        args.model_type,
        args.dataset,
        args.data_path,
        args.max_seqlen,
        stopwords_file=args.stopwords_file,
        has_tm=args.model_type != "LSTM_LM",
        sentence_level=args.sentence_level,
        num_docs=args.num_docs,
        min_freq=args.vocab_min_freq,
        min_tm_freq=args.vocab_min_tm_freq,
        min_tm_doc_freq=args.vocab_min_tm_doc_freq,
        max_tm_freq_pct=args.vocab_max_tm_freq_pct,
        ignore_symbols=args.ignore_symbols,
        tm_sequence_length=args.tm_seqlen if args.model_type == "TDLM" else None,
        tm_doc_length=args.doc_len if args.model_type == "TDLM" else None,
    )
    batched = create_lm_batched(
        train_data, args.batch_size, ignore_keys=["cum_bows", "cum_tm_docs", "tm_sequences", "tm_docs"]
    )
    if args.model_type == "TDLM":
        batched["tm_docs"] = train_data["tm_docs"]
        batched["tm_sequences"] = train_data["tm_sequences"]
    # Reshape evaluation data to have bsz=1
    val_data["lm_sequences"] = val_data["lm_sequences"].reshape((-1, 1))
    test_data["lm_sequences"] = test_data["lm_sequences"].reshape((-1, 1))

    model_cf, writer_cf = make_config(args, len(vocab), num_tm_words, writer=(args.writer_type is not None))

    if args.writer_type == "wandb":
        run = wandb.init(project="[project]", entity="[entity]", config=writer_cf, name=args.run_name)
        metric_names = log_metric_names[args.model_type]
        for metric_name in metric_names:
            run.define_metric("train/" + metric_name, step_metric="epoch")
            run.define_metric("valid/" + metric_name, step_metric="epoch")
        writer = wandb
    elif args.writer_type == "tensorboard":
        writer_dir = "runs" if args.writer_dir is None else args.writer_dir
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.model_type}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        writer = SummaryWriter(log_dir=f"{writer_dir}/{run_name}")
    else:
        print("Not logging to wandb or tensorboard.")
        writer = None

    model = getattr(models, args.model_type)(**model_cf).to(device)
    num_lm_params, num_tm_params = model.get_num_parameters()
    print(
        f"(non emb) Num LM params: {num_lm_params[1]}, num TM params: {num_tm_params[1]}, total: {num_lm_params[1] + num_tm_params[1]}"
    )
    print(
        f"(inc emb) Num LM params: {sum(num_lm_params)}, num TM params: {sum(num_tm_params)}, total: {sum(num_lm_params) + sum(num_tm_params)}"
    )

    if args.pretrained_emb_file is not None:
        print(f"Loading pretrained embeddings from {args.pretrained_emb_file}...")
        init_embeddings(model, vocab, args.pretrained_emb_file, vocab[PAD])

    print(
        f"Training {args.model_type} for {args.num_epochs} epochs with {model.vocab_size} LM words and {num_tm_words} TM words."
    )

    best_val_ppl = float("inf")
    best_model_state = None
    patience_count = 0
    start_epoch = 1
    end_epoch = args.num_epochs + 1

    if args.model_type == "LSTM_LM":
        lm_parameters = model.parameters()
        tm_optimizer = None
    else:
        lm_parameters = model.lm_parameters()
        if args.model_type == "TDLM" and args.lm_only:
            tm_optimizer = None
        else:
            tm_optimizer = torch.optim.AdamW(
                model.tm_parameters() + model.stopword_parameters(), lr=args.tm_lr, weight_decay=args.wdecay
            )

    if args.lm_optimizer == "adam":
        lm_optimizer = torch.optim.AdamW(lm_parameters, lr=args.lm_lr, weight_decay=args.wdecay)
        scheduler = None
    elif args.lm_optimizer == "sgd":
        lm_optimizer = torch.optim.SGD(lm_parameters, lr=args.lm_lr, weight_decay=args.wdecay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            lm_optimizer,
            patience=0,
            factor=0.25,
            threshold=args.stopping_threshold,
            threshold_mode="abs",
            verbose=True,
        )

    try:
        for epoch in range(start_epoch, end_epoch):
            epoch_start_time = time.time()

            if args.warmup_kl is not None and args.warmup_kl > 0:
                kl_weight = args.kl_start + (1 - args.kl_start) * min(epoch, args.warmup_kl) / args.warmup_kl
            else:
                kl_weight = args.kl_start

            train(
                model,
                epoch,
                batched,
                args.max_seqlen,
                vocab[PAD],
                lm_optimizer,
                tm_optimizer,
                writer,
                kl_weight=kl_weight,
                tm_batch_size=args.tm_batch_size,
            )
            val_ppl, val_tm_ppl = evaluate(
                model, epoch, val_data, args.max_seqlen, vocab[PAD], writer, tm_batch_size=args.tm_batch_size
            )

            if args.model_type != "LSTM_LM" and epoch % 10 == 0:
                log_topics(
                    writer, get_topics_top_words(model.get_topics(), vocab, topn=20, to_print=True), epoch, n_topics=10
                )

            print("-" * 99)
            print(
                f"| end of epoch {epoch:3d} | time {(time.time() - epoch_start_time):5.2f}s | val ppl {val_ppl:8.2f}"
            )

            if val_ppl + args.stopping_threshold >= best_val_ppl:
                patience_count += 1
                print(f"Val loss > best val loss, not saving...")
            else:
                patience_count = 0
                best_val_ppl = val_ppl
                best_model_state = deepcopy(model.state_dict())

            if args.patience > 0 and patience_count >= args.patience:
                print("-" * 99)
                print(f"Didn't beat best val loss for {args.patience} epochs, stopping training.")
                break

            if scheduler is not None:
                scheduler.step(val_ppl)
                log(writer, "train", "lm_lr", scheduler._last_lr[0], epoch)
            else:
                log(writer, "train", "lm_lr", args.lm_lr, epoch)

    except KeyboardInterrupt:
        print("-" * 99)
        print("Exiting from training early")

    model.load_state_dict(best_model_state)
    test_ppl, test_tm_ppl = evaluate(
        model, None, test_data, args.max_seqlen, vocab[PAD], writer, tm_batch_size=args.tm_batch_size
    )

    print("=" * 99)
    print(f"End of training | tst ppl {test_ppl:8.2f} | best val ppl {best_val_ppl:8.2f}")
    print("=" * 99)

    if args.model_type != "LSTM_LM":
        log_topics(writer, get_topics_top_words(model.get_topics(), vocab, topn=20, to_print=True), None)

    if args.writer_type == "tensorboard":
        writer.add_hparams(writer_cf, {"valid/ppl": best_val_ppl})

    if args.save_dir is not None:
        print(f"Saving best model and vocab to {args.save_dir}")

        checkpoint = {
            "epoch": args.num_epochs,
            "model_type": args.model_type,
            "model_cf": model_cf,
            "model": best_model_state,
            "lm_optimizer": lm_optimizer.state_dict(),
            "tm_optimizer": tm_optimizer.state_dict() if tm_optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "vocab": vocab,
            "seed": args.seed,
            "wandb_id": wandb.run.id if args.writer_type == "wandb" else None,
        }

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        torch.save(checkpoint, os.path.join(args.save_dir, "final_ckpt.pt"))
        if args.model_type != "LSTM_LM":
            get_topics_top_words(
                model.get_topics(), vocab, topn=20, fname=os.path.join(args.save_dir, "topics_top20.txt")
            )


if __name__ == "__main__":
    main()

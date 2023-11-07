"""
Utility functions.
"""
import gensim.models as g
import math
import numpy as np
import os
import torch
import random
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

supported_model_types = {"LSTM_LM", "TDLM", "TRNN", "VRTM"}
supported_datasets = {"apnews", "imdb", "bnc", "wikitext-2"}
supported_optimizers = {"adam", "sgd"}
log_metric_names = {
    "LSTM_LM": ["ppl"],
    "TDLM": ["ppl", "tm_ppl"],
    "TRNN": ["ppl", "loss", "token_loss", "stopword_loss", "kl_loss"],
    "VRTM": ["ppl", "loss", "token_loss", "stopword_loss", "kl_loss"],
}
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"
NWL = "<nwl>"
dataset_to_dummy_symbols = {
    "apnews": [SOS, EOS, PAD, UNK],
    "imdb": [SOS, EOS, PAD, UNK],
    "bnc": [SOS, EOS, PAD, UNK],
    "wikitext-2": [EOS, PAD, UNK, NWL],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def get_topics_top_words(topic_weights, vocab, topn=10, to_print=False, fname=None):
    if isinstance(topic_weights, np.ndarray):
        word_indices = np.argsort(topic_weights)[:, ::-1]
    else:
        _, word_indices = topic_weights.sort(descending=True)

    topic_top_words = []

    for indices in word_indices[:, :topn]:
        topic_top_words.append(vocab.lookup_tokens(indices.tolist()))

    if to_print:
        for i, words in enumerate(topic_top_words):
            print(f"{i + 1}: {' '.join(words)}")

    if fname is not None:
        with open(fname, "w") as f:
            for i, words in enumerate(topic_top_words):
                f.write(" ".join(words) + "\n")

    return topic_top_words


def counter_to_lower(counter):
    new_counter = Counter()

    for k, v in counter.items():
        new_counter[k.lower()] += v

    return new_counter


def repackage_hidden(hidden, is_last):
    """Wraps hidden states in new Tensors, to detach them from their history.
       Reset hidden state following last sequence in each document.
    Args:
      hidden: The hidden states, with shape [2, num_layers, batch_size, hidden_dim].
      is_last: A list of ints indicating whether the last sequence of the document, with length batch_size.
    """
    if isinstance(hidden, torch.Tensor):
        is_last = np.array(is_last, dtype=bool)
        hidden[:, is_last, :] = 0

        return h.detach()
    else:
        return tuple(repackage_hidden(v, is_last) for v in hidden)


def init_embeddings(model, vocab, pretrained_emb_file, pad_idx):
    kvs = g.KeyedVectors.load_word2vec_format(pretrained_emb_file, binary=True)
    embeddings = []
    count = 0

    for token in vocab.get_itos():
        if token in kvs:
            count += 1
            embeddings.append(kvs[token])
        else:
            limit = 0.5 / kvs.vector_size
            embeddings.append(
                np.random.uniform(
                    -limit,
                    limit,
                    [
                        kvs.vector_size,
                    ],
                )
            )

    print(f"# tokens found: {count}/{len(vocab)}")
    device = next(model.parameters()).device
    model.init_pretrained(torch.tensor(np.array(embeddings), dtype=torch.float).to(device), pad_idx)


def tdlm_shuffle_lm_tm_batch_idxs(num_lm_batches, num_tm_batches):
    num_batches = num_lm_batches + num_tm_batches
    lm_batch_idxs = [("lm", i) for i in range(num_lm_batches)]
    tm_batch_idxs = [("tm", i) for i in range(num_tm_batches)]
    np.random.shuffle(tm_batch_idxs)
    batch_idxs = lm_batch_idxs
    batch_idxs = []
    lm_ptr, tm_ptr = 0, 0

    while lm_ptr < num_lm_batches and tm_ptr < num_tm_batches:
        if np.random.binomial(1, num_lm_batches / num_batches) == 0:
            batch_idxs.append(tm_batch_idxs[tm_ptr])
            tm_ptr += 1
        else:
            batch_idxs.append(lm_batch_idxs[lm_ptr])
            lm_ptr += 1

    batch_idxs += tm_batch_idxs[tm_ptr:]
    batch_idxs += lm_batch_idxs[lm_ptr:]

    return batch_idxs


def make_config(args, vocab_size, num_tm_words, writer=False):
    model_cf = dict(
        rnn_type="LSTM",
        vocab_size=vocab_size,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
    )

    if args.model_type != "LSTM_LM":
        model_cf.update(
            dict(
                num_topics=args.num_topics,
                num_tm_words=num_tm_words,
            )
        )
    if args.model_type == "TRNN":
        model_cf.update(
            dict(
                hidden_dim_theta=args.hidden_dim_theta,
                use_vrtm_stopword_net=args.use_vrtm_stopword_net,
            )
        )
    elif args.model_type == "VRTM":
        model_cf["seqlen"] = args.max_seqlen
    elif args.model_type == "TDLM":
        model_cf.update(
            dict(
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                topic_embedding_size=args.topic_embedding_size,
                tm_keep_prob=args.tm_keep_prob,
                lm_keep_prob=args.lm_keep_prob,
                lm_only=args.lm_only,
            )
        )

    if not writer:
        return model_cf, None

    writer_cf = dict(model_cf)
    writer_params = [
        "dataset",
        "model_type",
        "batch_size",
        "max_seqlen",
        "num_epochs",
        "patience",
        "seed",
        "sentence_level",
        "wdecay",
        "lm_optimizer",
        "stopping_threshold",
        "vocab_min_freq",
        "lm_lr",
    ]

    if args.model_type != "LSTM_LM":
        writer_params.extend(
            ["tm_lr", "vocab_min_tm_freq", "vocab_min_tm_doc_freq", "vocab_max_tm_freq_pct", "ignore_symbols"]
        )
    if args.model_type == "TRNN" or args.model_type == "VRTM":
        writer_params.extend(["warmup_kl", "kl_start"])
    elif args.model_type == "TDLM":
        writer_params.extend(["doc_len", "tm_batch_size", "tm_seqlen"])

    writer_cf.update({param: getattr(args, param) for param in writer_params})
    writer_cf.update(
        dict(
            pretrained_emb_file=os.path.basename(args.pretrained_emb_file)
            if args.pretrained_emb_file is not None
            else None,
            stopwords_file=os.path.basename(args.stopwords_file) if args.stopwords_file is not None else None,
        )
    )

    return model_cf, writer_cf


def set_other_model_type_args_to_none(args):
    if args.model_type != "TRNN" and args.model_type != "VRTM":
        args.hidden_dim_theta = None
        args.warmup_kl = None
        args.kl_start = None
    if args.model_type != "TRNN":
        args.use_vrtm_stopword_net = None
    if args.model_type != "TDLM":
        args.tm_batch_size = None
        args.filter_sizes = None
        args.num_filters = None
        args.topic_embedding_size = None
        args.tm_keep_prob = None
        args.lm_keep_prob = None
        args.tm_seqlen = None
        args.doc_len = None
    if args.model_type == "LSTM_LM":
        args.tm_lr = None


def log(writer, split, name, value, epoch):
    if type(writer) == SummaryWriter:
        writer.add_scalar(f"{split}/{name}", value, epoch)
    elif writer is not None:
        writer.log({f"{split}/{name}": value, "epoch": epoch})


def log_dict(writer, split, dict, epoch):
    if type(writer) == SummaryWriter:
        for key, value in dict.items():
            log(writer, split, key, value, epoch)
    elif writer is not None:
        wandb_dict = {f"{split}/{k}": v for k, v in dict.items()}
        wandb_dict["epoch"] = epoch
        writer.log(wandb_dict)


def log_topics(writer, topics, epoch, n_topics=None):
    if writer is None or type(writer) == SummaryWriter:
        return

    from wandb import Table

    key = f"topics_{epoch}e" if epoch is not None else "topics_final"
    topics = [" ".join(topic) for topic in topics]

    if n_topics is not None:
        topics = topics[:n_topics]

    topics_and_idx = [[i, item] for i, item in enumerate(topics)]

    writer.log({key: Table(columns=["idx", "top_words"], data=topics_and_idx)})

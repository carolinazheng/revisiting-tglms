"""
Data processing and batching for TGLM training/testing.
"""
import codecs
import numpy as np
import os
import re
import time
import torch
import torchtext.vocab
from collections import Counter, OrderedDict

from utils import PAD, UNK, SOS, EOS, counter_to_lower, dataset_to_dummy_symbols


def get_data_iter(data_path, split="train"):
    """Get the documents for a given split of a dataset, and word counters.
    Each line is a document, already tokenized.
    - In APNEWS, IMDB, and BNC, sentence boundaries are demarcated by "\t".
    - In WikiText-2, explicit newlines in the data are represented by <nwl> tokens.
    """
    fname = os.path.join(data_path, "{}.txt".format(split))
    docs = []
    word_counter = Counter()
    word_doc_counter = Counter()

    for line in codecs.open(fname, "r", "utf-8"):
        docs.append(line)
        doc_words = set()

        if split == "train":
            for word in line.strip().split():
                word_counter[word] += 1
                doc_words.add(word)
            for word in doc_words:
                word_doc_counter[word] += 1

    if split == "train":
        return docs, word_counter, word_doc_counter
    else:
        return docs


def create_vocab(
    has_tm,
    counter,
    doc_counter,
    stopwords,
    dummy_symbols,
    min_freq=10,
    min_tm_freq=0,
    min_tm_doc_freq=0,
    max_tm_freq_pct=0.001,
    ignore_symbols=True,
):
    """Create LM and TM vocabularies."""
    # Create LM vocab: Remove tokens below min_freq and add dummy symbols
    filtered = {}

    for token, freq in counter.items():
        if freq >= min_freq:
            filtered[token] = freq

    filtered = Counter(filtered)

    for token in dummy_symbols:
        filtered[token] = 1

    # Create TM vocab: Additionally remove certain tokens
    if has_tm:
        counter_lower = counter_to_lower(counter)
        doc_counter_lower = counter_to_lower(doc_counter)

        stopwords = set([stopword for stopword in stopwords])
        num_freqwords = int(len(counter_lower) * max_tm_freq_pct)
        freqwords = set([item[0] for item in counter_lower.most_common(num_freqwords)])
        rarewords = set([token for token, freq in counter_lower.items() if freq < min_tm_freq])
        rarewords_doc = {item for item, doc_freq in doc_counter_lower.items() if doc_freq < min_tm_doc_freq}

        ignore = stopwords | set(dummy_symbols) | freqwords | rarewords | rarewords_doc

        if ignore_symbols:
            alpha_check = re.compile("[a-zA-Z]")
            symbols = set(
                [token for token in filtered.keys() if (alpha_check.search(token) is None or token.startswith("'"))]
            )
            ignore |= symbols | set(["n't"])

        ignore = {token for token in filtered if token.lower() in ignore}
    else:
        ignore = filtered.keys()

    filtered_ordered = OrderedDict(filtered)
    num_tm_words = len(filtered_ordered) - len(ignore)

    # Move non-TM words to the end of vocab
    for token in ignore:
        if token in filtered_ordered:
            filtered_ordered.move_to_end(token)
        else:
            filtered_ordered[token] = 1

    # Sort TM/non-TM partitions of vocab by frequency
    items = list(filtered_ordered.items())
    sorted_tm = sorted(items[:num_tm_words], key=lambda x: (x[1], x[0]), reverse=True)
    sorted_ignore = sorted(items[num_tm_words:], key=lambda x: (x[1], x[0]), reverse=True)
    final_ordered = OrderedDict()

    for k, v in sorted_tm:
        final_ordered[k] = v
    for k, v in sorted_ignore:
        final_ordered[k] = v

    # Create final vocab object
    vocab = torchtext.vocab.vocab(final_ordered)
    vocab.set_default_index(vocab.lookup_indices([UNK])[0])

    return vocab, len(vocab) - len(ignore)


def create_sequence_context_data(
    model_type,
    dataset,
    docs,
    vocab,
    num_tm_words,
    lm_sequence_length,
    tm_sequence_length,
    tm_doc_length,
    sentence_level,
):
    """Create the LM sequences and the document context used by the TGLMs for each sequence."""
    start = time.time()

    if model_type == "VRTM" or model_type == "TRNN":
        sequence_context_data = create_topic_biased_lm_sequence_context_data(
            dataset, docs, vocab, num_tm_words, lm_sequence_length, sentence_level, create_bows=(model_type == "TRNN")
        )
    elif model_type == "TDLM":
        sequence_context_data = create_joint_tm_lm_sequence_context_data(
            dataset, docs, vocab, num_tm_words, lm_sequence_length, tm_sequence_length, tm_doc_length, sentence_level
        )
    elif model_type == "LSTM_LM":
        sequence_context_data = create_topic_biased_lm_sequence_context_data(
            dataset, docs, vocab, num_tm_words, lm_sequence_length, sentence_level, create_bows=False
        )

    print(
        f"Finished creating data: {len(sequence_context_data['lm_sequences'])} sequences and {len(docs)} docs in {time.time() - start:.2f} seconds."
    )

    return sequence_context_data


def create_joint_tm_lm_sequence_context_data(
    dataset, docs, vocab, num_tm_words, lm_sequence_length, tm_sequence_length, tm_doc_length, sentence_level
):
    lm_sequences = []
    tm_sequences = []
    cum_tm_docs = []  # only contains previous TM words in doc; len(lm_sequences)
    all_tm_docs = []  # contains all TM words in doc minus current seq; len(lm_sequences)
    tm_docs = []  # contains all TM words in doc; len(docs)
    is_last = []

    for doc_idx, doc in enumerate(docs):
        # 1) Add special tokens
        if dataset.startswith("wikitext"):
            doc = EOS + " " + doc.strip() + " " + EOS
        else:  # is a tdlm dataset (apnews, imdb, or bnc)
            doc = SOS + " " + doc.strip() + " " + EOS

        # 2) Remove sentence-demarcating tabs
        if not dataset.startswith("wikitext"):
            doc = doc.replace("\t", " \t " if sentence_level else " ")

        # 3) Convert to list of tokens
        doc = doc.split(" ")

        # 4) Create tm_doc representation and TM sequences (seq, doc_idx)
        def pad_or_clamp(sequence, pad_token, length):
            if len(sequence) > length:
                return sequence[:length]
            while len(sequence) < length:
                sequence.append(pad_token)
            return sequence

        tm_doc = []
        tm_sequence = []

        for word in doc:
            token = vocab[word]

            if token < num_tm_words:
                # Add to TM sequence
                tm_sequence.append(token)
                if len(tm_sequence) == tm_sequence_length:
                    tm_sequences.append((tm_sequence, doc_idx))
                    tm_sequence = []
                # Add to TM doc vector
                if len(tm_doc) < tm_doc_length + lm_sequence_length:
                    tm_doc.append(token)

        tm_doc = pad_or_clamp(tm_doc, vocab[PAD], tm_doc_length + lm_sequence_length)
        tm_docs.append(tm_doc[:tm_doc_length])

        # 5) Create LM sequences that will be formed into batches
        def finish_sequence(tm_doc_idx, seq_is_last=False):
            lm_sequences.append(np.array(sequence))
            seq_num_tm_words = len([token for token in sequence if token < num_tm_words])
            cum_tm_docs.append(np.array(pad_or_clamp(tm_doc[:tm_doc_idx], vocab[PAD], tm_doc_length)))
            all_tm_docs.append(
                np.array(
                    pad_or_clamp(
                        tm_doc[:tm_doc_idx] + tm_doc[tm_doc_idx + seq_num_tm_words :], vocab[PAD], tm_doc_length
                    )
                )
            )
            is_last.append(1 if seq_is_last else 0)
            return [], tm_doc_idx + seq_num_tm_words

        sequence = []
        tm_doc_idx = 0

        for word in doc:
            if len(sequence) == lm_sequence_length:
                sequence, tm_doc_idx = finish_sequence(tm_doc_idx)
            if (sentence_level and word == "\t") or word == EOS:
                if word == EOS:
                    sequence.append(vocab[EOS])
                while len(sequence) < lm_sequence_length:
                    sequence.append(vocab[PAD])
                sequence, tm_doc_idx = finish_sequence(tm_doc_idx, seq_is_last=True)
            else:
                sequence.append(vocab[word])

    return dict(
        lm_sequences=np.array(lm_sequences),
        is_last=np.array(is_last),
        cum_tm_docs=np.array(cum_tm_docs),
        all_tm_docs=np.array(all_tm_docs),
        tm_docs=np.array(tm_docs),
        tm_sequences=tm_sequences,
    )


def create_topic_biased_lm_sequence_context_data(
    dataset, docs, vocab, num_tm_words, sequence_length, sentence_level, create_bows=True
):
    sequences = []
    cum_bows = []
    all_bows = []
    is_last = []
    doc_lens = []

    for idx, doc in enumerate(docs):
        # 1) Add special tokens
        if dataset.startswith("wikitext"):
            doc = EOS + " " + doc.strip() + " " + EOS
        else:  # is a tdlm dataset (apnews, imdb, or bnc)
            doc = SOS + " " + doc.strip() + " " + EOS

        # 2) Remove sentence-demarcating tabs
        if not dataset.startswith("wikitext"):
            doc = doc.replace("\t", " \t " if sentence_level else " ")
        elif sentence_level:
            doc = doc.replace(".", "\t")

        # 3) Convert to list of tokens
        doc = doc.split(" ")

        # 4) Compute document length
        doc_len = len(doc) - 1

        if sentence_level:
            doc_len -= sum([1 if word == "\t" else 0 for word in doc])

        # 5) Create tm bag-of-words representation
        doc_bows = np.zeros(num_tm_words, dtype=np.int32)

        for word in doc:
            token = vocab[word]
            if token < num_tm_words:
                doc_bows[token] += 1

        # 6) Iterate over tokens and create the sequence-context tuples based on BPTT length
        # For each sequence-context tuple:
        # - sequence is the input to the lm
        # - cum_bows, all_bows are the context
        # - is_last is a binary indicator of whether the sequence is the last in the document
        # - doc_len is the length of the document, needed for the loss function
        def finish_sequence(prev_bows, reset_bows=False):
            sequences.append(np.array(sequence))
            cum_bows.append(prev_bows.copy())
            all_bows.append(doc_bows.copy())
            is_last.append(1 if reset_bows else 0)
            doc_lens.append(doc_len)

            if reset_bows:
                prev_bows = np.zeros(num_tm_words, dtype=np.int32)
            else:
                for token in sequence:
                    if token < num_tm_words:
                        prev_bows[token] += 1

            return [], prev_bows

        prev_bows = np.zeros(num_tm_words, dtype=np.int32)
        sequence = []

        for i, word in enumerate(doc):
            if len(sequence) == sequence_length:
                sequence, prev_bows = finish_sequence(prev_bows)
            if (sentence_level and word == "\t") or word == EOS:
                if word == EOS:
                    sequence.append(vocab[EOS])
                while len(sequence) < sequence_length:
                    sequence.append(vocab[PAD])
                sequence, prev_bows = finish_sequence(prev_bows, reset_bows=True)
            else:
                sequence.append(vocab[word])

    data = dict(
        lm_sequences=np.array(sequences),
        is_last=np.array(is_last),
        cum_bows=np.array(cum_bows),
        all_bows=np.array(all_bows),
        doc_lens=np.array(doc_lens),
    )

    if not create_bows:
        del data["cum_bows"], data["all_bows"]

    return data


def process_dataset(
    model_type,
    dataset,
    data_path,
    lm_sequence_length,
    stopwords_file=None,
    has_tm=True,
    sentence_level=False,
    num_docs=None,
    testing=False,
    vocab=None,
    num_tm_words=None,
    compute_bows=False,
    min_freq=10,
    min_tm_freq=0,
    min_tm_doc_freq=0,
    max_tm_freq_pct=0.001,
    ignore_symbols=True,
    tm_sequence_length=None,
    tm_doc_length=None,
):
    """Main function to create the vocabulary and data. For testing, pass in a vocab and num_tm_words."""
    if testing:
        test_docs = get_data_iter(data_path, split="test")
        test_data = create_sequence_context_data(
            model_type,
            dataset,
            test_docs[:num_docs],
            vocab,
            num_tm_words,
            lm_sequence_length,
            tm_sequence_length,
            tm_doc_length,
            sentence_level,
        )

        return None, None, test_data, vocab, num_tm_words

    if stopwords_file is not None:
        with open(stopwords_file, "r") as f:
            stopwords = set(f.read().splitlines())
    else:
        stopwords = {}

    train_docs, word_counter, word_doc_counter = get_data_iter(data_path, split="train")
    val_docs = get_data_iter(data_path, split="valid")
    test_docs = get_data_iter(data_path, split="test")

    if vocab is None:
        vocab, num_tm_words = create_vocab(
            has_tm,
            word_counter,
            word_doc_counter,
            stopwords,
            dataset_to_dummy_symbols[dataset],
            min_freq=min_freq,
            min_tm_freq=min_tm_freq,
            min_tm_doc_freq=min_tm_doc_freq,
            max_tm_freq_pct=max_tm_freq_pct,
            ignore_symbols=ignore_symbols,
        )

    train_data = create_sequence_context_data(
        model_type,
        dataset,
        train_docs[:num_docs],
        vocab,
        num_tm_words,
        lm_sequence_length,
        tm_sequence_length,
        tm_doc_length,
        sentence_level,
    )
    val_data = create_sequence_context_data(
        model_type,
        dataset,
        val_docs[:num_docs],
        vocab,
        num_tm_words,
        lm_sequence_length,
        tm_sequence_length,
        tm_doc_length,
        sentence_level,
    )
    test_data = create_sequence_context_data(
        model_type,
        dataset,
        test_docs[:num_docs],
        vocab,
        num_tm_words,
        lm_sequence_length,
        tm_sequence_length,
        tm_doc_length,
        sentence_level,
    )

    return train_data, val_data, test_data, vocab, num_tm_words


# Use this for train batches only -- eval/test with batch_size = 1; sequences.reshape((-1, 1))
def create_lm_batched(data_dict, batch_size, ignore_keys=set()):
    sequences = data_dict["lm_sequences"]
    max_seqlen = len(sequences[0])
    sequences = sequences.reshape(-1)
    n_extra_tokens = len(sequences) % (batch_size * max_seqlen)
    last_target_token = sequences[-n_extra_tokens]
    sequences = sequences[:-n_extra_tokens]  # chop off a few extra tokens
    batched_sequences = sequences.reshape((-1, batch_size), order="F")
    final_row = np.append(batched_sequences[0][1:], last_target_token)  # for target tokens in the last batch
    batched_sequences = np.vstack([batched_sequences, final_row])

    print(f"Created {len(batched_sequences) // max_seqlen} batches.")

    def create_context_batched(data, batch_size):
        if len(data.shape) == 1:
            return data.reshape((-1, batch_size), order="F").reshape(-1)
        else:
            y_shape = data.shape[-1]
            return data.reshape((-1, batch_size, y_shape), order="F").reshape((-1, y_shape))

    batched = dict(lm_sequences=batched_sequences)
    n_extra_sequences = n_extra_tokens // max_seqlen

    for key, value in data_dict.items():
        if key != "lm_sequences" and key not in ignore_keys:
            batched[key] = create_context_batched(value[:-n_extra_sequences], batch_size)

    return batched


def get_lm_batch(batched_data, batch_idx, max_seqlen, device):
    batch_size = batched_data["lm_sequences"].shape[1]
    batch = dict(
        data=batched_data["lm_sequences"][batch_idx * max_seqlen : (batch_idx + 1) * max_seqlen],
        target=batched_data["lm_sequences"][batch_idx * max_seqlen + 1 : (batch_idx + 1) * max_seqlen + 1].reshape(-1),
    )

    for key, value in batched_data.items():
        if key != "lm_sequences" and not key.startswith("tm_"):
            batch[key] = value[batch_idx * batch_size : (batch_idx + 1) * batch_size]

    for key, value in batch.items():
        batch[key] = torch.tensor(value, device=device)

    return batch


def get_tm_batch(batched_data, batch_idx, batch_size, device):
    cur_tm_sequences = batched_data["tm_sequences"][batch_idx * batch_size : (batch_idx + 1) * batch_size]
    data = np.vstack([batched_data["tm_docs"][doc_idx] for _, doc_idx in cur_tm_sequences])
    target = np.vstack([seq for seq, _ in cur_tm_sequences]).reshape(-1)

    return dict(
        data=torch.tensor(data, device=device),
        target=torch.tensor(target, device=device),
    )

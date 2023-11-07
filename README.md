# Revisiting Topic-Guided Language Models
This repo contains source code for the paper:

```
Revisting Topic-Guided Language Models
Carolina Zheng, Keyon Vafa, David M. Blei
TMLR 2023
```

## Requirements
Our Python version is `3.10`. The required packages can be installed inside a virtual environment via

`pip install -r requirements.txt` 

For training LDA via Mallet, install gensim 3.8.3 in a separate environment. (Newer versions of gensim do not support Mallet.)

Training runs can be logged using either Tensorboard or Weights and Biases (we used v0.15) via the `--writer_type` argument. To use W\&B, replace the project and entity arguments in `wandb.init` to use those of your own account.

## Datasets
The tokenized APNEWS, IMDB, and BNC datasets used in the experiments is available [here](https://github.com/mmrezaee/VRTM/#data-format).

To download and preprocess WikiText-2, install HuggingFace Datasets (we used v2.13.1) and run `python scripts/preprocess_wikitext.py --out_dir [save dir]`.

The stop word list and pre-trained embeddings are also available:
- [Mallet stop word list](https://github.com/mimno/Mallet/blob/master/stoplists/en.txt)
- [Google News pre-trained embeddings via word2vec](https://code.google.com/archive/p/word2vec/)

## Usage
For the below commands, if the command contains `...` at the end, there are additional optional arguments. Run `python [script filename] --help` to see a full list of arguments and their descriptions.

Supported dataset names are: apnews, imdb, bnc, or wikitext-2.

### Training

To train an LSTM-LM or a TGLM:

`python train.py --dataset [dataset name] --data_path [data splits dir] --model_type [LSTM_LM, TRNN, VRTM, or TDLM] ...`


### Testing
Given a model checkpoint, to run evaluation and/or save a TGLM's learned topics to a text file:

`python test.py --load_dir [checkpoint dir] ...`

### LDA
To train LDA:

`python lda/lda_mallet.py --dataset [dataset name] --data_path [data splits dir] --mallet_bin_path [Mallet binary path] ...`

### Coherence
To compute coherence after training a TGLM, we first need to preprocess the reference corpus given a topic model vocabulary.

Convert the vocab saved with the model checkpoint to a pickle used in the subsequent scripts:

`python coherence/convert_vocab.py --model_dir [checkpoint dir] --out_path [save path]`

Given a tokenized reference corpus, remove words not in the topic model vocabulary:

`python coherence/tokenize_texts_coherence.py --data_dir [data splits dir] --vocab_path [vocab path] --out_dirname [save dir]`

Then compute coherence:

`python coherence/compute_coherence.py --topics_path [topics text file path] --vocab_path [vocab path] --texts_dir [tokenized texts dir] --top_n [5, 10, 15, 20] --coherence_type c_npmi --window_size 10 --to_lower [1 for WT-2, 0 otherwise] ...`

### Probing
After training an LSTM-LM and a TGLM, here is how to run the probing pipeline.

First, run evaluation on a model checkpoint and save the hidden states (and topic proportions vectors, if a TGLM):

`python probe/create_probe_data.py --model_dir [checkpoint dir] --dataset [dataset name] --data_path [data splits dir] --out_dir [created data save dir] ...`

Then, given created data from an LSTM-LM and a TGLM, merge the data (LSTM-LM hidden states) and targets (topic proportion vectors):

`python probe/merge_probe_data.py --data_dir_tglm [TGLM created data path] --data_dir_lstm_lm [LSTM-LM created data path] --out_dir [merged data save dir] --target_name [attention for TDLM, mu_theta for TopicRNN]`

Then train the probe on the merged data:

`python probe/train_probe.py --data_dir [merged data dir] --inverse_softmax [1 for TDLM, 0 for TopicRNN] --num_epochs 50 --lr 3e-5 --wdecay 1e-4 --patience 3 ...`

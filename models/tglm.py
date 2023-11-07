"""
The base class for all topic-guided language models.
"""
import torch.nn as nn


class TopicGuidedLanguageModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_topics,
        num_tm_words,
        dropout=0.0,
        tie_weights=False,
    ):
        super(TopicGuidedLanguageModel, self).__init__()
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_topics = num_topics
        self.num_tm_words = num_tm_words

        self.lm_param_prefixes = {"rnn", "encoder", "decoder"}
        self.lm_embed_param_prefixes = {"encoder", "decoder"}
        # These prefixes are defined by the TGLM subclass model
        self.stopword_param_prefixes = set()
        self.tm_param_prefixes = set()
        self.tm_embed_param_prefixes = set()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        rnn_dropout = 0 if num_layers == 1 else dropout

        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(embed_dim, hidden_dim, num_layers, dropout=rnn_dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, nonlinearity=nonlinearity, dropout=rnn_dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            if hidden_dim != embed_dim:
                raise ValueError("When using the tied flag, hidden_dim must be equal to emsize")
            self.decoder.weight = self.encoder.weight

    def init_weights(self):
        """
        Randomly initialize the embedding weights.
        """
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_pretrained(self, embeddings, pad_idx):
        """
        Initialize the embedding weights with pretrained embeddings.
        """
        self.encoder = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=pad_idx)

    def init_hidden(self, bsz):
        """
        Set the RNN hidden state to zero.
        """
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.num_layers, bsz, self.hidden_dim),
                weight.new_zeros(self.num_layers, bsz, self.hidden_dim),
            )
        else:
            return weight.new_zeros(self.num_layers, bsz, self.hidden_dim)

    def lm_parameters(self):
        """
        Returns: A list of language model parameters.
        """
        return [param for name, param in self.named_parameters() if name.split(".")[0] in self.lm_param_prefixes]

    def tm_parameters(self):
        """
        Returns: A list of topic model parameters.
        """
        return [param for name, param in self.named_parameters() if name.split(".")[0] in self.tm_param_prefixes]

    def stopword_parameters(self):
        """
        Returns: A list of stopword model parameters.
        """
        return [param for name, param in self.named_parameters() if name.split(".")[0] in self.stopword_param_prefixes]

    def get_num_parameters(self):
        """
        Returns:
          num_lm_params: A tuple of (# LM embedding parameters, # LM other parameters).
          num_tm_params: A tuple of (# TM embedding parameters, # TM other parameters).
        """
        num_lm_params = [0, 0]
        num_tm_params = [0, 0]

        for name, params in self.named_parameters():
            prefix = name.split(".")[0]

            if prefix in self.lm_param_prefixes or prefix in self.stopword_param_prefixes:
                if prefix in self.lm_embed_param_prefixes:
                    num_lm_params[0] += params.numel()
                else:
                    num_lm_params[1] += params.numel()
            elif prefix in self.tm_param_prefixes:
                if prefix in self.tm_embed_param_prefixes:
                    num_tm_params[0] += params.numel()
                else:
                    num_tm_params[1] += params.numel()
            else:
                raise Exception(f"Unknown parameter prefix: {prefix}")

        return tuple(num_lm_params), tuple(num_tm_params)

    def get_topics(self):
        """
        Returns: The topic-word distribution matrix, with shape [num_topics, num_tm_words].
        """
        raise NotImplementedError()

    def forward(self):
        """
        Args: The data and the initial hidden state.
        Returns: A dict of loss terms and the final hidden state.
        """
        raise NotImplementedError()

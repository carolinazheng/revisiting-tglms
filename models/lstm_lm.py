"""
This code is adapted from the PyTorch examples codebase, which implements the model in Zaremba et al., 2014: Recurrent Neural Network Regularization.
https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
        tie_weights=False,
    ):
        super(LSTMLanguageModel, self).__init__()
        self.model_type = "LSTM_LM"
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_param_prefixes = {"encoder", "decoder"}

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

        self.init_weights()

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

    def get_num_parameters(self):
        """
        Returns:
          A tuple of (# LM embedding parameters, # LM other parameters).
          A tuple of (0, 0).
        """
        num_params = [0, 0]

        for name, params in self.named_parameters():
            if name.split(".")[0] in self.embed_param_prefixes:
                num_params[0] += params.numel()
            else:
                num_params[1] += params.numel()

        return tuple(num_params), (0, 0)

    def forward(self, input, hidden):
        """Compute a forward pass through the model.

        Args:
          input: The sequence tokens, with shape [seqlen, batch_size].
          hidden: The initial hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        Returns:
            The logits for the next token prediction, with shape [seqlen * batch_size, vocab_size].
            hidden: The final hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

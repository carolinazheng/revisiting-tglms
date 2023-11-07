"""
Our implementation of the model in Rezaee and Ferarro, 2020: A Discrete Variational Recurrent Topic Model without the Reparametrization Trick.
The paper's codebase: https://github.com/mmrezaee/VRTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Dirichlet
from typing import Dict, Tuple, Union

from .tglm import TopicGuidedLanguageModel

epsilon = torch.finfo(torch.float32).eps


def _batched_index_select(input, dim, index):
    """
    Args:
      input: The input tensor, with shape [B, *].
      dim: The dimension to select from.
      index: The indices to select, with shape [B, M].
    Returns:
      The selected elements, with shape [B, M, *].
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.gather(input, dim, index)


class EncoderGamma(nn.Module):
    def __init__(self, seqlen, embed_dim, num_topics):
        super().__init__()
        self.W_gamma = nn.Parameter(torch.randn(seqlen, embed_dim, num_topics))
        self.seqlen = seqlen

    def forward(self, input):
        """
        Args:
          input: The sequence topic model embeddings, with shape [seqlen, batch_size, embed_dim].
        Returns:
          The document-topic variational distribution parameters, with shape [batch_size, num_topics].
        """
        if input.size(0) < self.seqlen:
            # Add padding
            input = torch.cat(
                (input, torch.zeros((self.seqlen - input.size(0), input.size(1), input.size(2)), device=input.device))
            )

        return torch.tensordot(input, self.W_gamma, ([0, 2], [0, 1]))


class BatchNormSwap(nn.Module):
    def __init__(self, c):
        super().__init__()
        self._batch_norm = nn.BatchNorm1d(c)

    def forward(self, input):
        """
        Args:
          input: The input tensor, with shape [*, batch_size, c].
        Returns:
          The batch-normalized tensor, with shape [*, batch_size, c].
        """
        return self._batch_norm(input.movedim(0, 2)).movedim(2, 0)


class VariationalRecurrentTopicModel(TopicGuidedLanguageModel):
    def __init__(
        self,
        rnn_type,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_topics,
        num_tm_words,
        seqlen,
        dropout=0.0,
        tie_weights=False,
    ):
        super(VariationalRecurrentTopicModel, self).__init__(
            rnn_type, vocab_size, embed_dim, hidden_dim, num_layers, num_topics, num_tm_words, dropout, tie_weights
        )
        self.model_type = "VRTM"
        self.seqlen = seqlen
        self.tm_param_prefixes = {"encoder_phi", "encoder_gamma", "topics"}
        self.tm_embed_param_prefixes = {"topics"}
        self.stopword_param_prefixes = {"encoder_stopword", "decoder_stopword"}

        self.encoder_stopword = nn.Sequential(
            nn.Linear(hidden_dim, 300),
            nn.Softplus(),
            nn.Linear(300, 50),
            nn.Softplus(),
        )
        self.decoder_stopword = nn.Sequential(
            nn.Linear(50, 1),
            BatchNormSwap(1),
        )
        self.topics = nn.Sequential(
            nn.Linear(num_tm_words, num_topics, bias=False),
            BatchNormSwap(num_topics),
            nn.Softplus(),
        )
        self.encoder_phi = nn.Sequential(
            nn.Linear(embed_dim, num_topics),
            BatchNormSwap(num_topics),
            nn.Softmax(dim=-1),
        )
        self.encoder_gamma = nn.Sequential(
            EncoderGamma(seqlen, embed_dim, num_topics),
            nn.Softplus(),
        )
        self.init_weights()

    def init_weights(self):
        """
        Randomly initialize the embedding and encoder gamma weights.
        """
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(next(self.decoder.parameters()), -initrange, initrange)
        nn.init.uniform_(next(self.encoder_gamma.parameters()), -initrange, initrange)

    def get_topics(self):
        """
        Returns:
          topics: The topic-word distribution matrix, with shape [num_topics, num_tm_words].
        """
        topics = next(self.topics.parameters()).detach()
        return topics

    def forward(self, input, target, hidden):
        """Compute a forward pass through the model.

        Args:
          input: The sequence tokens, with shape [seqlen, batch_size].
          target: The target sequence tokens, with shape [seqlen, batch_size].
          hidden: The RNN initial hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        Returns:
          A dict of the outputs used to compute the loss, with items:
            stopword_logits: The stopword prediction logits, with shape [seqlen * batch_size].
            rnn_logits: The token prediction logits, with shape [seqlen, batch_size, vocab_size].
            topics: The topic-word distribution matrix, with shape [num_topics, num_tm_words].
            q_phis: The parameters of the topic-word variational distributions, with shape [seqlen, batch_size, num_topics].
            q_gammas: The parameters of the document-topic variational distributions, with shape [batch_size, num_topics].
            q_thetas_sampled: The sampled document-topic vectors, with shape [batch_size, num_topics].
          hidden: The RNN final hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        rnn_logits = self.decoder(output)  # seqlen x vocab_size

        target = target.view(input.shape)
        counts_by_vocab = F.one_hot(target).sum(0)
        counts_by_vocab[:, self.num_tm_words :] = 0
        counts = _batched_index_select(counts_by_vocab, 1, target.T).T
        emb_tm = counts.unsqueeze(-1) * emb
        q_phi = self.encoder_phi(emb_tm)  # seqlen x num_topics
        q_gamma = self.encoder_gamma(emb_tm) + epsilon  # num_topics
        q_theta_dist = Dirichlet(q_gamma)
        q_theta = q_theta_dist.rsample() if self.training else q_gamma / q_gamma.sum(-1).unsqueeze(-1)
        stopword_logits = self.decoder_stopword(self.encoder_stopword(output)).squeeze().reshape(-1)

        return (
            dict(
                stopword_logits=stopword_logits,
                rnn_logits=rnn_logits,
                topics=next(self.topics.parameters()),
                q_phis=q_phi,
                q_gammas=q_gamma,
                q_thetas_sampled=q_theta,
            ),
            hidden,
        )

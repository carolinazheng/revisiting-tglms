"""
Our implementation of the model in Dieng et al., 2017: TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from .tglm import TopicGuidedLanguageModel


class TopicRNN(TopicGuidedLanguageModel):
    def __init__(
        self,
        rnn_type,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_topics,
        num_tm_words,
        hidden_dim_theta,
        dropout=0.0,
        tie_weights=False,
        use_vrtm_stopword_net=True,
    ):
        super(TopicRNN, self).__init__(
            rnn_type, vocab_size, embed_dim, hidden_dim, num_layers, num_topics, num_tm_words, dropout, tie_weights
        )
        self.model_type = "TRNN"
        self.tm_param_prefixes = {"embed_theta", "encoder_theta", "decoder_theta", "mu_theta", "logsigma_theta"}
        self.stopword_param_prefixes = {"encoder_stopword", "decoder_stopword"}
        self.tm_embed_param_prefixes = {"embed_theta", "decoder_theta"}

        self.embed_theta = nn.Linear(num_tm_words, hidden_dim_theta)
        self.encoder_theta = nn.Sequential(
            self.embed_theta,
            nn.ReLU(),
            nn.Linear(hidden_dim_theta, hidden_dim_theta),
            nn.ReLU(),
        )
        self.mu_theta = nn.Linear(hidden_dim_theta, num_topics)
        self.logsigma_theta = nn.Linear(hidden_dim_theta, num_topics)
        self.decoder_theta = nn.Linear(num_topics, num_tm_words, bias=False)

        if use_vrtm_stopword_net:
            self.encoder_stopword = nn.Sequential(
                nn.Linear(hidden_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 50),
                nn.ReLU(),
            )
            self.decoder_stopword = nn.Linear(50, 1)
        else:
            self.encoder_stopword = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.decoder_stopword = nn.Identity()

        self.init_weights()

    def get_topics(self):
        """
        Returns:
          topics: The topic-word distribution matrix, with shape [num_topics, num_tm_words].
        """
        topics = next(self.decoder_theta.parameters()).T.detach()
        return topics

    def _encode_rnn(self, input, hidden):
        """
        Args:
          input: The sequence tokens, with shape [seqlen, batch_size].
          hidden: The language model initial hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        Returns:
          output: The output of the RNN, with shape [seqlen, batch_size, hidden_dim].
          hidden: The language model final hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output, hidden

    def _normalize_bows(self, bows):
        """
        Args:
          bows: The bag-of-words representation of the document, with shape [batch_size, num_tm_words].
        Returns:
          The normalized bag-of-words representation of the document, with shape [batch_size, num_tm_words].
        """
        eps = torch.finfo(torch.float32).eps
        freq = (bows.sum(1) + eps).unsqueeze(1)
        return bows / freq

    def _encode_theta(self, bows):
        bows = self._normalize_bows(bows)
        q_theta = self.encoder_theta(bows)
        mu_theta = self.mu_theta(q_theta)
        logsigma_theta = self.logsigma_theta(q_theta)
        return mu_theta, logsigma_theta

    def forward(self, input, bows, hidden):
        """Compute a forward pass through the model.

        Args:
          input: The sequence tokens, with shape [seqlen, batch_size].
          bows: The bag-of-words representation of the document, with shape [batch_size, num_tm_words].
          hidden: The RNN initial hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        Returns:
          A dict of the outputs used to compute the loss, with items:
            stopword_output: The stopword prediction logits, with shape [seqlen * batch_size].
            token_rnn_output: The token prediction logits from the RNN, with shape [seqlen * batch_size, vocab_size].
            token_mixed_output: The token prediction logits from the mixed model, with shape [seqlen * batch_size, vocab_size].
            mu_theta: The mean of the document-topic variational distributions, with shape [batch_size, num_topics].
            logsigma_theta: The log standard deviation of the document-topic variational distributions, with shape [batch_size, num_topics].
          hidden: The RNN final hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        """
        output, hidden = self._encode_rnn(input, hidden)
        rnn_logits = self.decoder(output)
        dist_mask = torch.zeros_like(rnn_logits)
        dist_mask[:, :, : self.num_tm_words] = float("-inf")
        rnn_output = F.log_softmax(rnn_logits + dist_mask, dim=-1).view(-1, self.vocab_size)

        mu_theta, logsigma_theta = self._encode_theta(bows)
        q_theta_dist = Normal(loc=mu_theta, scale=(0.5 * logsigma_theta).exp())
        theta = q_theta_dist.rsample() if self.training else mu_theta
        theta_logits = F.pad(self.decoder_theta(theta), (0, self.vocab_size - self.num_tm_words))
        theta_logits = theta_logits.expand(len(input), -1, -1)
        dist_mask = torch.zeros_like(rnn_logits)
        dist_mask[:, :, self.num_tm_words :] = float("-inf")
        mixed_output = F.log_softmax(rnn_logits + theta_logits + dist_mask, dim=-1).view(-1, self.vocab_size)
        stopword_output = self.decoder_stopword(self.encoder_stopword(output)).view(-1)

        return (
            dict(
                stopword_output=stopword_output,
                token_rnn_output=rnn_output,
                token_mixed_output=mixed_output,
                mu_theta=mu_theta,
                logsigma_theta=logsigma_theta,
            ),
            hidden,
        )

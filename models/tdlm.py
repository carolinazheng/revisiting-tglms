"""
Our implementation of the model in Lau et al., 2017: Topically Driven Neural Language Model.
The paper's codebase: https://github.com/jhlau/topically-driven-language-model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tglm import TopicGuidedLanguageModel


class GatingUnit(nn.Module):
    def __init__(self, hidden_dim, topic_embedding_size):
        super(GatingUnit, self).__init__()
        self.W_z = nn.Linear(topic_embedding_size, hidden_dim, bias=False)
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(topic_embedding_size, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(topic_embedding_size, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, rnn_output, mean_topic):
        """
        Args:
          rnn_output: The RNN output, with shape [seqlen * batch_size, hidden_dim].
          mean_topic: The topic model mean topic vector, with shape [batch_size, topic_embedding_size].
        Returns:
          The coupled output, with shape [seqlen * batch_size, hidden_dim].
        """
        z = self.sig(self.W_z(mean_topic) + self.U_z(rnn_output))
        r = self.sig(self.W_r(mean_topic) + self.U_r(rnn_output))
        c = self.tanh(self.W_h(mean_topic) + self.U_h(r * rnn_output))
        return (1 - z) * rnn_output + z * c


class TopicallyDrivenLanguageModel(TopicGuidedLanguageModel):
    def __init__(
        self,
        rnn_type,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_topics,
        num_tm_words,
        filter_sizes,
        num_filters,
        topic_embedding_size,
        tm_keep_prob,
        lm_keep_prob,
        dropout=0.0,
        tie_weights=False,
        lm_only=False,
    ):
        super(TopicallyDrivenLanguageModel, self).__init__(
            rnn_type, vocab_size, embed_dim, hidden_dim, num_layers, num_topics, num_tm_words, dropout, tie_weights
        )
        self.model_type = "TDLM"
        self.filter_sizes = filter_sizes
        self.filter_number = num_filters
        self.topic_embedding_size = topic_embedding_size
        self.conv_size = len(self.filter_sizes) * self.filter_number
        self.tm_keep_prob = tm_keep_prob
        self.lm_keep_prob = lm_keep_prob
        self.lm_only = lm_only

        self.lm_param_prefixes |= {"gate"}
        self.tm_param_prefixes = {
            "conv_word_embedding",
            "topic_output_embedding",
            "topic_input_embedding",
            "conv",
            "tm_softmax",
        }
        self.tm_embed_param_prefixes = {"conv_word_embedding", "tm_softmax"}

        self.conv_word_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.topic_output_embedding = nn.Linear(self.num_topics, self.topic_embedding_size)
        self.topic_input_embedding = nn.Linear(self.conv_size, self.num_topics)
        self.tm_softmax = nn.Linear(self.topic_embedding_size, num_tm_words)
        # note -- following naming from https://github.com/jhlau/topically-driven-language-model/blob/master/tdlm_config.py
        self.tm_dropout = nn.Dropout(self.tm_keep_prob)
        self.lm_dropout = nn.Dropout(self.lm_keep_prob)
        self.conv = nn.Conv2d(1, self.filter_number, (1, self.embed_dim), stride=1)
        self.gate = GatingUnit(hidden_dim, topic_embedding_size)

        self.init_weights()

    def get_topics(self):
        """
        Returns:
          topics: The topic-word distribution matrix, with shape [num_topics, num_tm_words].
        """
        eye = torch.eye(self.num_topics, device=next(self.parameters()).device)
        topics = self.tm_softmax(self.topic_output_embedding(eye)).detach()
        return topics

    def forward(self, input, hidden, document_context, mode, return_attention=False):
        """Compute a forward pass through the model.

        Args:
          input: The sequence tokens, with shape [seqlen, batch_size].
          hidden: The RNN initial hidden states, with shape [2, num_layers, batch_size, hidden_dim].
          document_context: The language model document context, with shape [batch_size, doc_len].
          mode: "lm" if it is a language model batch, "tm" if it is a topic model batch.
          return_attention: Whether to return the attention weights instead of the logits, for probing.
        Returns:
          logits: The logits for next token prediction, with shape [seqlen * batch_size, vocab_size].
          hidden: The RNN final hidden states, with shape [2, num_layers, batch_size, hidden_dim].
        """
        if mode not in {"lm", "tm"}:
            raise Exception(f"Unknown mode: {mode}")

        doc_inputs = self.conv_word_embedding(document_context)
        doc_inputs = self.tm_dropout(doc_inputs)[:, :, :, None]
        doc_inputs = torch.transpose(doc_inputs, 3, 1)
        doc_inputs = torch.transpose(doc_inputs, 2, 3)

        conv = self.conv(doc_inputs)
        h = torch.max(conv, 2)[0]
        pooled_outputs = [h]
        conv_pooled = torch.cat(pooled_outputs, 1)
        conv_pooled = torch.reshape(conv_pooled, [-1, self.conv_size])
        attention = self.topic_input_embedding(conv_pooled).softmax(-1)
        mean_topic = self.topic_output_embedding(attention)
        mean_topic = self.lm_dropout(mean_topic)

        if mode == "tm":
            tm_logits = self.tm_softmax(mean_topic)
            return F.log_softmax(tm_logits, dim=1)

        mean_topic = torch.tile(mean_topic, (len(input), 1))
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        coupled_output = self.gate(output.reshape(mean_topic.size(0), -1), mean_topic)
        decoded = self.decoder(coupled_output)
        logits = F.log_softmax(decoded, dim=1)

        if return_attention:
            return attention, hidden
        else:
            return logits, hidden

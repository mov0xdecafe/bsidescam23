import torch.nn.functional as ActFunc
from torch import nn

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, kernel_size, featureset_size):
        super().__init__()
        self._kernel_size = kernel_size
        self._hidden_dim = hidden_dim
        self._word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._conv = nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, embedding_dim))
        self._hidden2feature = nn.linear(hidden_dim, featureset_size)
    
    def forward(self, sample):
        embeds = self._word_embeddings(sample)
        # convert the vector so Conv2d can understand it
        conv_in = embeds.view(1, 1, len(sample), -1)
        conv_out = self._conv(conv_in)
        conv_out = ActFunc.relu(conv_out)
        hidden_in = conv_out.view(self._hidden_dim, len(sample) + 1 - self._kernel_size).transpose(0,1)
        feature_space = self._hidden2feature(hidden_in)
        feature_scores = ActFunc.log_softmax(feature_space, dim=1)

        return feature_scores
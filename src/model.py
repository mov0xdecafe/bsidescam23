import torch.nn.functional as F
from torch import nn

class CNNModel(nn.Module):
    """
    Convolutional Neural Network (CNN) model for feature extraction.

    Args:
        vocab_size (int): Vocabulary size for word embeddings.
        embedding_dim (int): Dimension of word embeddings.
        hidden_dim (int): Number of hidden units in convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
        featureset_size (int): Size of the output feature set.

    Attributes:
        _kernel_size (int): Size of the convolutional kernel.
        _hidden_dim (int): Number of hidden units in convolutional layer.
        _word_embeddings (nn.Embedding): Word embedding layer.
        _conv (nn.Conv2d): Convolutional layer.
        _hidden2feature (nn.Linear): Linear transformation to produce feature scores.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, kernel_size, featureset_size):
        super().__init__()
        self._kernel_size = kernel_size
        self._hidden_dim = hidden_dim
        self._word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._conv = nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, embedding_dim))
        self._hidden2feature = nn.Linear(hidden_dim, featureset_size)
    
    def forward(self, sample):
        """
        Forward pass through the CNN model.

        Args:
            sample (Tensor): Input data tensor.

        Returns:
            Tensor: Output tensor containing feature scores.
        """
        embeds = self._word_embeddings(sample)
        # Convert the vector so Conv2d can understand it
        conv_in = embeds.view(1, 1, len(sample), -1)
        conv_out = self._conv(conv_in)
        conv_out = F.relu(conv_out)
        hidden_in = conv_out.view(self._hidden_dim, len(sample) + 1 - self._kernel_size).transpose(0, 1)
        feature_space = self._hidden2feature(hidden_in)
        feature_scores = F.log_softmax(feature_space, dim=1)

        return feature_scores

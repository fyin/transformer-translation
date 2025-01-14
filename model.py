import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
   A class that represents input embeddings for a neural network model.

   This class initializes an embedding layer that maps vocabulary indices to dense vectors.
   It is commonly used in natural language processing tasks to convert words into numerical representations.

   Attributes:
       embedding_dim (int): The dimensionality of the embedding vector for each token.
       vocab_size (int): The size of the vocabulary (number of unique tokens). Fine tune the value to determine the optimal vocabulary size for the dataset.
       embedding (nn.Embedding): The embedding layer that transforms input indices into dense vectors.
       embedding (nn.Embedding): The embedding layer that transforms input indices into dense vectors.

   Methods:
       forward(x):
           Computes the forward pass by applying the embedding layer to the input indices
           and scaling the output by the square root of the embedding dimension.
   """
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor containing indices of the words in the vocabulary.

        Returns:
            Tensor: The resulting tensor after applying the embedding layer and scaling.
        """
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    A class that implements positional encoding for sequence data in neural networks.

    Positional encoding is used to inject information about the position of elements in a sequence,
    which is essential for models like Transformers that do not have a built-in notion of order.

    Attributes:
        embedding_dim (int): The dimensionality of the embedding vectors.
        seq_len (int): The length of the input sequences.
        dropout (nn.Dropout): A dropout layer to prevent overfitting.
        pe (Tensor): The precomputed positional encodings for the input sequences.

    Methods:
        forward(x):
            Applies the positional encoding to the input tensor and applies dropout.
    """

    def __init__(self, embedding_dim: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0)/embedding_dim))  # (embedding_dim / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / embedding_dim))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / embedding_dim))
        pe = pe.unsqueeze(0) # add a new dimension for batch, leading to pe tensor with (1, seq_len, embedding_dim)
        # Register the positional encoding as a buffer,
        # so that it is not considered a model parameter and will not be updated during backpropagation.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Tensor: The resulting tensor after adding positional encodings and applying dropout.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    A class that implements Layer Normalization for neural network layers.

    Layer Normalization is a technique used to normalize the inputs across the features,
    which helps stabilize and accelerate the training of deep learning models.

    Attributes:
        eps (float): A small constant added to the denominator for numerical stability.
        alpha (nn.Parameter): A learnable scale parameter for normalization.
        bias (nn.Parameter): A learnable bias parameter for normalization.

    Methods:
        forward(x):
            Applies layer normalization to the input tensor.
    """

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # Learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    A class that implements a feed-forward neural network block.

    This block consists of two linear transformations with a ReLU activation function
    in between, along with dropout for regularization. It is commonly used in transformer architectures.

    Attributes:
        linear_1 (nn.Linear): The first linear transformation from input dimension to hidden dimension.
        dropout (nn.Dropout): A dropout layer to prevent overfitting.
        linear_2 (nn.Linear): The second linear transformation from hidden dimension back to input dimension.

    Methods:
        forward(x):
            Applies the feed-forward transformation to the input tensor.
    """

    def __init__(self, embedding_dim: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu_(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
    A class that implements a multi-head attention mechanism.

    Multi-head attention allows the model to jointly attend to information from different representation subspaces
    at different positions, which is essential for capturing complex relationships in the input data.

    Attributes:
        embedding_dim (int): The dimensionality of the input and output vectors.
        h (int): The number of attention heads.
        d_k (int): The dimensionality of each attention head (embedding_dim / h).
        w_q (nn.Linear): Linear transformation for the query.
        w_k (nn.Linear): Linear transformation for the key.
        w_v (nn.Linear): Linear transformation for the value.
        w_o (nn.Linear): Linear transformation for the output.
        dropout (nn.Dropout): A dropout layer to prevent overfitting.

    Methods:
        attention(query, key, value, mask, dropout):
            Computes the attention scores and applies them to the value tensor.
        forward(q, k, v, mask):
            Applies the multi-head attention mechanism to the input tensors.
    """

    def __init__(self, embedding_dim: int, h: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.h = h
        assert embedding_dim % h == 0, "embedding_dim is not divisible by h"

        self.d_k = embedding_dim // h
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, embedding_dim)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Hidden layer
        # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
   A class that implements a residual connection with layer normalization and dropout.

   Residual connections help in training deep neural networks by allowing gradients to flow through the network
   more easily. This class applies layer normalization to the input, processes it through a sublayer (e.g.,
   feedforward network or multi-head attention), and adds the original input back to the output.

   Attributes:
       dropout (nn.Dropout): A dropout layer to prevent overfitting.
       norm (LayerNormalization): A layer normalization instance to normalize the input.

   Methods:
       forward(x, sublayer):
           Applies the residual connection to the input tensor using the specified sublayer.
   """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    # sublayer: Part of the overall model architecture, such as feedforward network, multi-head attention block, any other transformation applied to x.
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
   A class that implements an encoder block for a transformer model.

   The encoder block consists of a multi-head self-attention mechanism followed by a feed-forward neural network,
   with residual connections and layer normalization applied at each step. This structure allows the model to
   effectively capture dependencies in the input data.

   Attributes:
       self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention mechanism.
       feed_forward_block (FeedForwardBlock): The feed-forward neural network.
       residual_connections (nn.ModuleList): A list of residual connection layers for normalization and dropout.

   Methods:
       forward(x, src_mask):
           Applies the encoder block to the input tensor.
   """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    """
    A class that implements the encoder component of a transformer model.

    The encoder consists of multiple EncoderBlocks, each of which applies self-attention and feed-forward
    transformations. The final output is normalized using layer normalization.

    Attributes:
        layers (nn.ModuleList): A list of encoder layers to process the input.
        norm (LayerNormalization): A layer normalization instance to normalize the output.

    Methods:
        forward(x, mask):
            Applies the encoder layers to the input tensor and normalizes the output.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    A class that implements a decoder block for a transformer model.

    The decoder block consists of a self-attention mechanism, a cross-attention mechanism, and a feed-forward
    neural network, with residual connections and layer normalization applied at each step. This structure allows
    the model to effectively generate output sequences based on the input from the encoder.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention mechanism for the decoder.
        cross_attention_block (MultiHeadAttentionBlock): The multi-head attention mechanism that attends to the encoder's output.
        feed_forward_block (FeedForwardBlock): The feed-forward neural network.
        residual_connections (nn.ModuleList): A list of residual connection layers for normalization and dropout.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Applies the decoder block to the input tensor using the encoder's output and masks.
    """

    def __init__(self, features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
    A class that implements the decoder component of a transformer model.

    The decoder consists of multiple DecoderBlocks, each of which applies self-attention, cross-attention and feed-forward
    transformations. The final output is normalized using layer normalization.

    Attributes:
        layers (nn.ModuleList): A list of decoder layers to process the input.
        norm (LayerNormalization): A layer normalization instance to normalize the output.

    Methods:
        forward(x, mask):
            Applies the decoder layers to the input tensor and normalizes the output.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
     A class that implements the projection layer for transformer model.

      Projection layer map the final decoder output to the vocabulary size for token prediction.

       Attributes:
           embedding_dim (int): The dimensionality of the embedding vectors.
           vocab_size (int): The size of the vocabulary (number of unique tokens).

       Methods:
           forward(x):
               Applies linear layer to the input tensor.
      """

    def __init__(self, embedding_dim, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    """
    A class that implements transformer model.

    It includes three major components of the model, encoder, decoder and projection layer.

    Attributes:
    encoder (Encoder): Encoder component.
    decoder (Decoder): Decoder component.
    src_embed (InputEmbeddings): Embedding layer for source input.
    tgt_embed (InputEmbeddings): Embedding layer for target input.
    src_pos (PositionalEncoding): Positional encoding layer for source input.
    tgt_pos (PositionalEncoding): Positional encoding layer for target input.
    projection_layer (ProjectionLayer): Projection layer.

    Methods:
    encode(src, src_mask):
       Create encoder layers.
    decode(encoder_output, src_mask, tgt, tgt_mask):
       Create decoder layers.
    project(x):
        Create projection layer.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, embedding_dim)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, embedding_dim)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, embedding_dim: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Create a transformer model instance and initialize its weights and biases using Xavier uniform initialization,
    which is  a weight initialization method that helps with training stability by keeping the scale of gradients roughly the same in all layers.
    """

    # Create the embedding layers
    src_embed = InputEmbeddings(embedding_dim, src_vocab_size)
    tgt_embed = InputEmbeddings(embedding_dim, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(embedding_dim, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(embedding_dim, tgt_seq_len, dropout)

    # Create the encoder blocks
    # N: #number of encoder blocks in the model,
    # Increasing N increases the model's capacity and depth, but it also increases computational cost
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(embedding_dim, h, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, d_ff, dropout)
        encoder_block = EncoderBlock(embedding_dim, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(embedding_dim, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(embedding_dim, h, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, d_ff, dropout)
        decoder_block = DecoderBlock(embedding_dim, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(embedding_dim, nn.ModuleList(encoder_blocks))
    decoder = Decoder(embedding_dim, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(embedding_dim, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


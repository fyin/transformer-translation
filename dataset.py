import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    """
    BilingualDataset is a PyTorch Dataset class for preparing bilingual data used in sequence-to-sequence models.

    Attributes:
        seq_len (int): The maximum sequence length for input and output sequences.
        ds (Dataset): A dataset containing the source and target language sentence pairs.
        tokenizer_src (Tokenizer): Tokenizer for the source language.
        tokenizer_tgt (Tokenizer): Tokenizer for the target language.
        src_lang (str): The source language key in the dataset.
        tgt_lang (str): The target language key in the dataset.
        sos_token (Tensor): Start-of-sequence token in the target tokenizer.
        eos_token (Tensor): End-of-sequence token in the target tokenizer.
        pad_token (Tensor): Padding token in the target tokenizer.

    Methods:
        __len__:
            Returns the total number of source-target sentence pairs in the dataset.

        __getitem__:
            Prepares data for a specific index in the dataset. The method tokenizes input sequences, adds start-of-sequence and end-of-sequence tokens,
            applies padding to the maximum sequence length, and returns a dictionary containing:
                - encoder_input (Tensor): Tokenized and padded source language sentence with start and end tokens.
                - decoder_input (Tensor): Tokenized and padded target language sentence with a start token.
                - encoder_mask (Tensor): Mask identifying non-padding tokens in encoder input.
                - decoder_mask (Tensor): Causal mask combined with padding mask for decoder input.
                - label (Tensor): Tokenized and padded target language sentence with an end token, used as the label for model training.
                - src_text (str): Original source language sentence.
                - tgt_text (str): Original target language sentence.
    """
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """"
        The __getitem__ method is used to fetch and preprocess the data for a given index.
        It retrieves the source and target language sentences from the dataset
        (like in https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-it),
        tokenizes them, and applies padding to ensure that all sequences have the same length.
        It then constructs the encoder input, decoder input, and label tensors,
        which are used for training.
        """

        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Get list of token ids of the src/target text
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Get number of padding to addd for each sequence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # src has sos and eos
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # tgt has sos

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add sos, eos, and padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    """
    Calculate the causal mask to prevent a model from accessing future information during training or inference.

    :param size: The size of the square causal mask to be generated.
    :return: A 2D Boolean tensor representing the causal mask, where elements above the main diagonal are masked (False),
    and the diagonal and below are unmasked (True).
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    return mask == 0
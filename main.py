import torch
from torch import nn

# hyperparameters
_block_size = 8
_batch_size = 4
max_iters = 3000
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 300
eval_iters = 200


def get_batch(data: torch.Tensor) -> torch.Tensor:
    ix = torch.randint(len(data) - _block_size, (_batch_size,))
    x = torch.stack([data[i : i + _block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + _block_size + 1] for i in ix])
    return x, y


def load_training(f: str = "input.txt") -> str:
    with open(f, "r") as f:
        return f.read()


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

def main():
    raw_training_data = load_training()

    # TODO(j.swannack): Look into other encoding schemes
    # i.e.
    #   sentencepiece
    #   byte pair encoding (tiktoken)

    # Character level tokeniser
    chars = sorted(set(raw_training_data))
    vocab_size = len(chars)

    # create mapping from string to int and vice versa
    s_to_i = {s: i for i, s in enumerate(chars)}
    i_to_s = {i: s for i, s in enumerate(chars)}
    encode = lambda x: [s_to_i[c] for c in x]
    decode = lambda x: "".join([i_to_s[i] for i in x])

    # # TODO(j.swannack): try this later
    # enc = tiktoken.get_encoding("gpt2")
    # encode = enc.encode
    # decode = enc.decode

    # encode the training data
    data = torch.tensor(encode(raw_training_data), dtype=torch.long)

    # get train and validation splits
    _n = int(len(data) * 0.9)
    train_data = data[:_n]
    val_data = data[_n:]


if __name__ == "__main__":
    import logging
    import ipdb

    logging.basicConfig(level=logging.INFO)

    with ipdb.launch_ipdb_on_exception():
        main()

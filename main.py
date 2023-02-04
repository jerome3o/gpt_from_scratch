import tiktoken
import torch

_block_size = 8
_batch_size = 4


def get_batch(data: torch.Tensor) -> torch.Tensor:
    ix = torch.randint(len(data) - _block_size, (_batch_size,))
    x = torch.stack([data[i : i + _block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + _block_size + 1] for i in ix])
    return x, y


def load_training(f: str = "input.txt") -> str:
    with open(f, "r") as f:
        return f.read()


def main():
    raw_training_data = load_training()

    # TODO(j.swannack): Look into other encoding schemes
    # i.e.
    #   sentencepiece
    #   byte pair encoding (tiktoken)

    # Character level tokeniser
    chars = sorted(set(raw_training_data))
    vocab_size = len(chars)
    s_to_i = {s: i for i, s in enumerate(chars)}
    i_to_s = {i: s for i, s in enumerate(chars)}
    encode = lambda x: [s_to_i[c] for c in x]
    decode = lambda x: "".join([i_to_s[i] for i in x])

    # # TODO(j.swannack): try this later
    # enc = tiktoken.get_encoding("gpt2")
    # encode = enc.encode
    # decode = enc.decode

    data = torch.tensor(encode(raw_training_data), dtype=torch.long)

    _n = int(len(data) * 0.9)
    train_data = data[:_n]
    val_data = data[_n:]

    xb, yb = get_batch(train_data)
    print(xb.shape, yb.shape)
    print(xb)
    print(yb)

    for batch_i in range(_batch_size):
        for block_i in range(_block_size):
            context = xb[batch_i, : block_i + 1]
            target = yb[batch_i, block_i]
            print(
                f"when the context is {decode(context.tolist())}, the target is {decode([target.tolist()])}"
            )


if __name__ == "__main__":
    import logging
    import ipdb

    logging.basicConfig(level=logging.INFO)

    with ipdb.launch_ipdb_on_exception():
        main()

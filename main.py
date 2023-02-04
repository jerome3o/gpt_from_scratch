import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters
_BLOCK_SIZE = 8
_BATCH_SIZE = 4
MAX_ITERS = 3000
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL = 300
EVAL_ITERS = 200


def get_batch(data: torch.Tensor) -> torch.Tensor:
    ix = torch.randint(len(data) - _BLOCK_SIZE, (_BATCH_SIZE,))
    x = torch.stack([data[i : i + _BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + _BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def load_training(f: str = "input.txt") -> str:
    with open(f, "r") as f:
        return f.read()


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:

        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = Non
        else:
            B, T, C = logits.shape
            # logits = logits.reshape(-1, C)  # (B*T, C)
            logits = logits.view(B * T, C)  # (B*T, C)
            targets = targets.view(B * T)  # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate a sequence of tokens from a starting token."""
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # get the predictions
                logits, _ = self.forward(idx)

                # get the last prediction
                logits = logits[:, -1, :]  # (B, C)

                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)

                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                # append the new token to the sequence
                idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

            return idx


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

    model = BigramLanguageModel(vocab_size).to(DEVICE)

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        # get the data
        xb, yb = get_batch(train_data)

        # get the predictions
        logits, loss = model(xb, yb)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get accurate loss prediction
        if iter % EVAL_INTERVAL == 0:
            # get the validation loss
            x, y = get_batch(val_data).to(DEVICE)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, val_loss = model(x, y)

            # get the training loss
            x, y = get_batch(train_data).to(DEVICE)
            _, train_loss = model(x, y)

            print(f"iter {iter}: train loss = {train_loss:.3f}, val loss = {val_loss:.3f}")


if __name__ == "__main__":
    import logging
    import ipdb

    logging.basicConfig(level=logging.INFO)

    with ipdb.launch_ipdb_on_exception():
        main()

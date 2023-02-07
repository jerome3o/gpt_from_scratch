import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL = 300
EVAL_ITERS = 200
N_EMBED = 32

print(DEVICE)


def get_batch(data: torch.Tensor) -> torch.Tensor:
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def load_training(f: str = "input.txt") -> str:
    with open(f, "r") as f:
        return f.read()


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> torch.Tensor:
    out = {}
    model.eval()
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size, max_block_size=BLOCK_SIZE):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(max_block_size, max_block_size)),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        w = q @ k.transpose(-2, -1) / self.head_size ** 0.5 

        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1) # (B, T, T)

        v = self.value(x) # (B, T, C)
        out = w @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """ simple linear layer with relu activation, using nn.Sequential"""
    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class MultiHead(nn.Module):
    """ A multi-head attention layer. """
    def __init__(self, n_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return concat of all heads
        return torch.cat([h(x) for h in self.heads], dim=-1)


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        self.sa_heads = MultiHead(n_heads=4, head_size=N_EMBED // 4)
        self.ff = FeedForward(N_EMBED)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=DEVICE)
        )  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.sa_heads(x) # (B, T, C)

        # feed forward
        x = self.ff(x) # (B, T, C)

        # get the logits
        logits = self.lm_head(token_embeddings)  # (B, T, vocab_size)

        if targets is None:
            loss = None
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
                # crop to block size
                idx_cond = idx[:, -BLOCK_SIZE:]  # (B, T)

                # get the predictions
                logits, _ = self.forward(idx_cond)

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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # get accurate loss prediction
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"iter: {iter} train {losses['train']:.3f} val {losses['val']:.3f}")

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    print(decode(model.generate(context, 1000).cpu().numpy()[0]))


if __name__ == "__main__":
    import logging
    import ipdb

    logging.basicConfig(level=logging.INFO)

    with ipdb.launch_ipdb_on_exception():
        main()

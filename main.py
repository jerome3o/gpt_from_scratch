import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
# MAX_ITERS = 5000
MAX_ITERS = 2
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBED = 384
N_HEAD = 6
N_LAYERS = 6
DROP_RATE = 0.2

print(DEVICE)
torch.manual_seed(1337)


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
        self.dropout = nn.Dropout(DROP_RATE)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # (B, T, C) @ (B, C, T) -> (B, T, T)
        w = q @ k.transpose(-2, -1) / self.head_size**0.5

        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)  # (B, T, T)
        w = self.dropout(w)

        v = self.value(x)  # (B, T, C)
        out = w @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """simple linear layer with relu activation, using nn.Sequential"""

    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROP_RATE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHead(nn.Module):
    """A multi-head attention layer."""

    def __init__(self, n_heads: int, head_size: int):
        super().__init__()
        n_embed = n_heads * head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(DROP_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return concat of all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """Transformer block.

    communication followed by computation + residual connection.
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHead(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEAD) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)

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
    print("vocab_size: ", vocab_size)

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

    print(train_data.shape, val_data.shape)

    model = Transformer(vocab_size).to(DEVICE)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        # get accurate loss prediction
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"iter: {iter} train {losses['train']:.3f} val {losses['val']:.3f}")

        # get the data
        xb, yb = get_batch(train_data)

        # get the predictions
        logits, loss = model(xb, yb)

        # backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    print(decode(model.generate(context, 1000).cpu().numpy()[0]))


if __name__ == "__main__":
    import logging
    import ipdb

    logging.basicConfig(level=logging.INFO)

    with ipdb.launch_ipdb_on_exception():
        main()

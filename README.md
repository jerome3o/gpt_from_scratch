# Let's build GPT: from scratch, in code, spelled out.

Code following along from [this youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Setup

Getting training data:
```sh
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Installing pytorch:

Go to the [pytorch site](https://pytorch.org/) and select the pip version applicable for you, this one if for ROCm5.2 as I am usually on AMD cards:

```sh
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
```

Install other dependencies:

```
pip install -r requirements.txt
pip install -r requirements.txt.dev
```

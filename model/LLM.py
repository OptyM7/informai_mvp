import math
from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import requests

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, quert, value, projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask,nut following the OpenAI/MF 
        # naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch 
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh = hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T,T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
        


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self,x ):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
         

class Block(nn.Module):
    
    def __init__ (self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 #  number of tokens: 50,000 BPE merges + 256 bytes tokens + 1
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimesnion

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # output embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # grey boxes
            ln_f = nn.LayerNorm(config.n_embd),  # linear box
        )) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

    def forward(self, idx):
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}. block size is."
        # forward the token and position embeddings
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

num_return_sequences = 5
max_length = 30

hi = GPTConfig()
model = GPT(hi)
model.eval()
#model.to

# prefix tokens, convert strings to token sequence
import tiktoken 
enc = tiktoken. get_encoding ("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (0,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to("cpu")


# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to (42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x,xcol), dim=1)

# print the generated text
for i in range (num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

import math
from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import requests
import tiktoken

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

    def forward(self, idx, target = None):
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
        loss = None
        if target != None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # flatten out the tensors to be 32 by 50257 and 31 by 1 respectively
            #cross entropy  -logn(probability)
        return logits, loss

#-------------------------------------------------------------------------------------------------
# select device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"~device used:{device}")


# randomly initialise model
model = GPT(GPTConfig())
model.eval()

# import dataset
with open("tinyshakespeare.txt", "r") as file: # the with statement is a keyword which is used to call 
    # __enter__ and __exit__  special methods in the class. These methods basically configure the environment of the class
    # the special environment is only available within the with block, exiting the with block the __exit__ method is called
    # and the environment is set back to default.
    data = file.read()[:1000]
enc = tiktoken.get_encoding('gpt2')
b, t = 4, 32  # batch and tokens
tokens = enc.encode(data)
tokens = tokens[:b*t+1]
x = torch.tensor(tokens[:-1]).view(b,t) # test data
y = torch.tensor(tokens[1:]).view(b,t)  # label data


logits, loss = model(x,y) # Strange that I'm directly calling class object of GPT with no method (e.g. model.method(x,y)
# This is because the superclass of model contains a special method __call__ which is called whenever you
#directly call a class object (its like init but for already existing class objects). 
# the __call__ method for nn.Module does a number of things, and one of them is to call the child classes
# forward() implementation, which it does in this case. 
# logits is a tensor of shape (B, T, vocab_size), and contains basically a score for every possible token in the token set
# for each token in x (a score predicting what out of the token should be next based on x's current token)
# if you call this in the no_grad with block it won't use gradient descent to develop the score.
# which basically removes some of the smartness of the LLM (doesn't use context)

print(loss) # should be around 1/50257 if the model was initialised properly
# # take the logits at the last position
# logits = logits[:, -1, :] # (B, vocab_size)
# # get the probabilities
# probs = F.softmax(logits, dim=-1) # make the logits into probabilities that add up to 1 sum is |1|1|... up to 32
# # do top-k sampling of 50 (huggingface pipeline default)
# # topk_probs here becomes (5, 50), topk_indices is (5, 50)
# topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
# # select a token from the top-k probabilities
# ix = torch.multinomial(topk_probs, 1) # (B, 1)
# # gather the corresponding indices
# xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
# # append to the sequence
# x = torch.cat((x,xcol), dim=1)

# # print the generated text
# for i in range (num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)

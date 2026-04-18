import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from dataclasses import dataclass
from contextlib import nullcontext
from tqdm.auto import tqdm
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
print(ds.shape)
encoding = tiktoken.get_encoding("gpt2") # uses tiktoken to get a tokenizing model

def processing(sample_text): #function to process text and convert to numbers for SLM to process.
    ids = encoding.encode_ordinary(sample_text['text']) 
    out = {'ids':ids,'len':len(ids)} #create a dictionary with your tokenized text and then the length after.
    return out

if not os.path.exists("train.bin"): #checks if we already have this step done, if we do then theres no point doing it again
    tokenized = ds.map(
        processing, # tokenize each row of wikitext-2
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
        ) #processes every row in the dataset
    for split, dset in tokenized.items(): #loops through each split in the dataset
        if split == 'test': #skips all of the test splits.
            continue
        arr_len = np.sum(dset['len'], dtype=np.uint64) #adds up token count across every row, so we know how big to make the file.
        filename = f'{split}.bin' #creates filenames.bin
        dtype = np.uint16 #tells you total # of bits 16 so 2^16 which is a big enough size. 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,)) #creates the file
        total_batches = 1024 #write data in 1024 chunks instaed of all at once
        idx = 0 #counter for where you are  as you write each chunk.
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'): #loops 1024 times, once per chunk
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy') #gets bactch number and converts it so numpy can use it
            arr_batch = np.concatenate(batch['ids']) #flattens all lists into a single array
            arr[idx : idx + len(arr_batch)] = arr_batch 
            idx += len(arr_batch) #moves position counter forward by how hamny tokens were written
        arr.flush() #saves everything to disk.

block_size = 128
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'

def get_batch(split): #gets random 32 chunks of 128 tokens from Disk and sends to GPU to be trained on. Loads and prepares data for forward propogation
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

class LayerNorm(nn.Module): #something important that is said in the notebook, I dont really understand its meaning.
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    


class CausalSelfAttention(nn.Module): #learns the importnace of specific words to other words. Like when someone says "it" can learn what does "it" refer to and when there are verbs it can see how nouns are important to the verb.
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 #checks that embedding isze is divisible by nyumber of heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) #linear layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) #
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)   #this is out logits matrix which i have explaine below...
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # the loss is cross entropy
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens given a conditioning sequence.
        idx: Tensor of shape (B, T)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

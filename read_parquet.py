# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:12:06 2024

@author: jim
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from tokenizers import Encoding
from transformers import AutoTokenizer

def read(file="str") -> pd.DataFrame():
    df = pd.read_parquet(file, engine='fastparquet')
    return df


df = read("train-00000-of-00001.parquet")

df = df.drop(columns=["instruction"])
df = df.dropna()
print(df.head())

eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tch_tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese")

df["input_token"] = df.apply(lambda row: eng_tokenizer(row['input']), axis=1)
df["output_token"] = df.apply(lambda row: tch_tokenizer(row['output']), axis=1)

empty_word = tch_tokenizer.encode("")
print(empty_word)

train, test = train_test_split(df, test_size=0.2, shuffle=True)
train.info()
test.info()

input_max = 256
output_max = 256

# for ti in train['input_token']:
#     if input_max < len(ti.input_ids):
#         input_max = len(ti.input_ids)
# for to in train['output_token']:
#     if output_max < len(to.input_ids):
#         output_max = len(to.input_ids)
# print(input_max)
# print(output_max)

print(train.head())
# print(train['input_token'][1340].tokens())
ix = torch.randint(len(train), (8,))

def get_batch(split):
    data = train if split == 'train' else test
    ix = torch.randint(len(data)-1, (batch_size,))
    x = []
    y_decoder_input = []
    y_target = []
    for i in ix:
        x1 = torch.tensor([data.iloc[i.item(), 2].input_ids])
        y1_decoder_input = torch.tensor([data.iloc[i.item(), 3].input_ids[:-1]])
        y1_target = torch.tensor([data.iloc[i.item(), 3].input_ids[1:]])
        x.append(F.pad(x1, pad = (0, input_max - x1.numel()), mode='constant', value=0))
        y_decoder_input.append(F.pad(y1_decoder_input, pad = (0, output_max - y1_decoder_input.numel()), mode='constant', value=0))
        y_target.append(F.pad(y1_target, pad = (0, output_max - y1_target.numel()), mode='constant', value=0))
    x = torch.stack(x).squeeze(1)
    y_decoder_input = torch.stack(y_decoder_input).squeeze(1)
    y_target = torch.stack(y_target).squeeze(1)
    x.to(device)
    y_decoder_input.to(device)
    y_target.to(device)
    return x, y_decoder_input, y_target

batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 50 # for estimated_loss which smooth the loss by averaging eval_interval number of loss
learning_rate = 1e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size_eng = eng_tokenizer.vocab_size
vocab_size_tch = tch_tokenizer.vocab_size

print(f"vocab_size of english = {vocab_size_eng}")
print(f"vocab_size of tchinese = {vocab_size_tch}")
# ---**---
torch.manual_seed(9527)

@torch.no_grad() # this telling PyTorch everything happens inside this function requires no grads hence requires no backward on
def estimated_loss():
    out = {}
    # model is set to evaluation phase
    # although there is no different on this model
    # however there are models that contains batch normalization, drop out layers which do make the difference
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y_d, Y_t = get_batch(split)
            logits, loss = m(X, Y_d, Y_t)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() # model is set to training phase
    return out

class MaskedHead(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        '''
         If you have parameters in your model,
         which should be saved and restored in the state_dict,
         but not trained by the optimizer,
         you should register them as buffers.
        '''
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # compute attention scores ("affinities")
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
      
        # perform weight aggregation of the value
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class Head(nn.Module):
    
    def __init__(self, head_size, encoder_x=None):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.encoder_x = encoder_x # check cross attention or not
        '''
         If you have parameters in your model,
         which should be saved and restored in the state_dict,
         but not trained by the optimizer,
         you should register them as buffers.
        '''
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_x=None): # k v is optional
        B,T,C = x.shape
        # compute attention scores ("affinities")
        q = self.query(x) # (B,T,C)
        k = None
        if encoder_x == None:
            k = self.key(x) # (B,T,C)
        else:
            k = self.key(encoder_x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
      
        # perform weight aggregation of the value.
        v = None
        if encoder_x == None:
            v = self.value(x) # (B,T,C)
        else:
            v = self.value(encoder_x) # (B,T,C)
            
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    

class MaskedMultiHeadAttention(nn.Module):
    '''multiple self-attention in parallel'''
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # dropout
    def forward(self, x, get_head=False):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        if get_head:
            return out, self.heads()
        return out

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # dropout
    def forward(self, x, encoder_x):
        x = torch.cat([h(x, encoder_x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out
    
class MultiHeadAttention(nn.Module):
    '''multiple self-attention in parallel'''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # dropout
    def forward(self, x, get_q=False, get_k=False, get_v=False):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # paper 3.3 in attention is all you need
        # d_model * 4 = d_ff
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

    

        
class DecoderBlock(nn.Module):

    def __init__(self, n_embd, n_head): # n_embd: embedding dimension, n_head: the number head we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa1 = MaskedMultiHeadAttention(n_head, head_size)
        self.sa2 = CrossMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layernorm added
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
    
    def forward(self, seq_input): # forward only allow one input
        x, encoder_output = seq_input[0], seq_input[1]
        # print("Decoder block x ", x.shape)
        x = x + self.sa1(self.ln1(x)) # residual connection
        x = x + self.sa2(self.ln2(x), encoder_output) # residual connection
        x = x + self.ffwd(self.ln3(x))
        # print("Decoder block x out ", x.shape)
        seq_output = (x, encoder_output)
        return seq_output

class EncoderBlock(nn.Module):

    def __init__(self, n_embd, n_head): # n_embd: embedding dimension, n_head: the number head we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layernorm added
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # print("Encoder block x ", x.shape)
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x))
        # print("Encoder block x out ", x.shape)
        return x
        
'''----------------------------------------------------------------------------------------------'''


class Decoder(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # each token directly read of the logits for the token from lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)

    self.block = nn.Sequential(*[DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)]) # Sequential not work with multiple input
    

  def forward(self, encoder_output, decoder_input): # target is optional
    B, T = decoder_input.shape
    # print(B, T)
    # idx and table are both [B,T] tensor of integers
    tok_embd = self.token_embedding_table(decoder_input) # [B,T,C]
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # [T,C]
    # print("Decoder t ", tok_embd.shape)
    # print("Decoder p ", pos_embd.shape)
    x = tok_embd + pos_embd # [B,T,C] x not only hold token identity, but also the position token occur.
    # print("Decoder x ", x.shape)
    seq_input = (x, encoder_output)
    # print("Decoder_pre packed x", seq_input[0].shape)
    x = self.block(seq_input) # apply one-head self-attention (B,T,C)
    return x[0]

    

    # in PyTorch cross_entropy if we have a multidimensional input, be sure that channel is in the second dimension
    
    

    
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly read of the logits for the token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
    
    def forward(self, idx): # target is optional
        # print("Encoder ", idx.shape)
        B, T = idx.shape
        # idx and table are both [B,T] tensor of integers

        tok_embd = self.token_embedding_table(idx) # [B,T,C]
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # [T,C]
        # print("Encoder t ", tok_embd.shape)
        # print("Encoder p ", pos_embd.shape)
        x = tok_embd + pos_embd # [B,T,C] x not only hold token identity, but also the position token occur.
        # print("Encoder x ", x.shape)
        x = self.block(x) # apply one-head self-attention (B,T,C)
        # print("Encoder x out ", x.shape)
      
        return x


class Transformers_Model(nn.Module):
    
    def __init__(self, encoder_vocab_size, decoder_vocab_size):
        super().__init__()
        self.encoder = Encoder(encoder_vocab_size).to(device)
        self.decoder = Decoder(decoder_vocab_size).to(device)
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, decoder_vocab_size)
        
    def forward(self, idx, decoder_input=None, targets=None):
        # print("Trans ",idx.shape)
        encoder_output = self.encoder(idx)
        x = self.decoder(encoder_output, decoder_input)
        x = self.ln(x)
        logits = self.lm_head(x) # [B,T,vocab_size]
        
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def translate(self, context):
        context = torch.tensor(eng_tokenizer.encode(context)).unsqueeze(0)
        decoder_current_input = [empty_word[0]] # <|SOS|>
        while True:
            encoder_output = self.encoder(context)
            x = self.decoder(encoder_output, torch.tensor(decoder_current_input).unsqueeze(0))
            x = self.ln(x)
            logits = self.lm_head(x) # [1,T,vocab_size]
            logits = logits[:, -1, :]
            logits = logits.squeeze(dim=0)
            probs = F.softmax(logits, dim=-1).squeeze()
            idx_next = torch.multinomial(probs, num_samples=1)
            decoder_current_input.append(idx_next.item())
            if idx_next == empty_word[1] or len(decoder_current_input)>=255:
                # print(decoder_current_input)
                return tch_tokenizer.decode(decoder_current_input, skip_special_tokens=True)
        

model = Transformers_Model(encoder_vocab_size=vocab_size_eng, decoder_vocab_size=vocab_size_tch)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val set
    if iter % eval_interval == 0:
        losses = estimated_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample on batch of data
    xb, y_decoder_inputb, y_targetb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb, y_decoder_inputb, y_targetb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = "I am curious about the real-world applications of synthetic data that you’ve actually used in your machine learning projects."
print(m.translate(context))




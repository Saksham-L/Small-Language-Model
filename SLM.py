import tiktoken
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
#!/bin/python

from GPTDataset import GPTDatasetV1
from torch.utils.data import DataLoader
import tiktoken

DATA_FILE = "../datasets/the-verdict.txt"


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    """
    Loads the data of the GPTDatasetV1 into a DataLoader
    """

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,  tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# Small text can read it all in one shot
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

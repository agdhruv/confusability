import os
from datasets import load_dataset, Dataset, load_from_disk
from itertools import islice


# ---- Streaming text -> fixed-length blocks (IterableDataset style) ----
def chunk_stream(ds, block_size: int, tokenizer):
    """
    Each chunk is a sequence of length block_size tokens.
    """
    buf = []
    for ex in ds:
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids: 
            continue
        buf.extend(ids)
        while len(buf) >= block_size:
            yield {"input_ids": buf[:block_size]}
            buf = buf[block_size:]

def load_train_dataset(
    train_ds_name: str,
    train_ds_config: str,
    datadir: str,
    tokenizer,
    block_size: int,
    num_blocks: int, # number of blocks (each of size block_size) to sample from the dataset
    split: str = "train",
):
    if not os.path.exists(datadir):
        print(f"Processed data not found. Creating and saving to {datadir}...")
        train_stream = load_dataset(train_ds_name, train_ds_config, split=split, streaming=True)
        train_iterable = chunk_stream(train_stream, block_size, tokenizer)
        materialized = list(islice(train_iterable, num_blocks))
        train_ds = Dataset.from_list(materialized)
        train_ds.save_to_disk(datadir)
    else:
        print(f"Loading processed data from {datadir}...")
        train_ds = load_from_disk(datadir)
    return train_ds

def load_eval_dataset(
    eval_ds_name: str,
    eval_ds_config: str,
    eval_datadir: str,
    tokenizer,
    block_size: int,
    num_blocks: int, # number of blocks (each of size block_size) to sample from the dataset
):
    if not os.path.exists(eval_datadir):
        print(f"Processed eval data not found. Creating and saving to {eval_datadir}...")
        eval_raw = load_dataset(eval_ds_name, eval_ds_config, split="test")
        eval_iterable = chunk_stream(eval_raw, block_size, tokenizer)
        materialized = list(islice(eval_iterable, num_blocks))
        eval_ds = Dataset.from_list(materialized)
        eval_ds.save_to_disk(eval_datadir)
    else:
        print(f"Loading processed eval data from {eval_datadir}...")
        eval_ds = load_from_disk(eval_datadir)
    return eval_ds
import os
from datasets import load_dataset, Dataset, load_from_disk
from itertools import islice


# ---- Streaming text -> fixed-length blocks (IterableDataset style) ----
def chunk_stream(ds, block_size: int, tokenizer):
    buf = []
    for ex in ds:
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids: 
            continue
        buf.extend(ids)
        while len(buf) >= block_size:
            yield {"input_ids": buf[:block_size]}
            buf = buf[block_size:]

def load_data_with_cache(
    datadir: str,
    block_size: int,
    tokenizer,
    num_samples: int,
    split: str = "train",
):
    if not os.path.exists(datadir):
        print(f"Processed data not found. Creating and saving to {datadir}...")
        train_stream = load_dataset("allenai/c4", "en", split=split, streaming=True)
        train_iterable = chunk_stream(train_stream, block_size, tokenizer)
        materialized = list(islice(train_iterable, num_samples))
        train_ds = Dataset.from_list(materialized)
        train_ds.save_to_disk(datadir)
    else:
        print(f"Loading processed data from {datadir}...")
        train_ds = load_from_disk(datadir)

def load_eval_dataset(
    eval_name: str,
    eval_config: str,
    tokenizer,
    block_size: int,
    num_samples: int,
):
    eval_raw = load_dataset(eval_name, eval_config, split="test")
    eval_iterable = chunk_stream(eval_raw, block_size, tokenizer)
    eval_ds: Dataset = Dataset.from_list(list(islice(eval_iterable, num_samples)))
    return eval_ds
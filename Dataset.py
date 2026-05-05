import torch
from torch.utils.data import Dataset, DataLoader
from TokenizationAndBPE import BPETokenizer

# Holds the full token stream and serves sliding windows as (input, target) pairs
class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], context_length: int):
        self.ids = token_ids
        self.ctx_len = context_length

    def __len__(self) -> int:
        return len(self.ids) - self.ctx_len

    def __getitem__(self, idx: int):
        chunk = self.ids[idx : idx + self.ctx_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input:  tokens 0..L-1
        y = torch.tensor(chunk[1:],  dtype=torch.long)  # target: tokens 1..L
        return x, y


def load_personal_data(directory: str) -> str:
    # Walk directory, find all .txt and .md files, join with end-of-text separator
    import os
    texts = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(('.txt', '.md')):
                path = os.path.join(root, fname)
                with open(path, encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
    return "\n<|endoftext|>\n".join(texts)


def load_hf_dataset_texts(dataset_name: str, split: str = "train", text_column: str = "text",
                           max_examples: int = None, min_chars: int = 0, template: str = None) -> list[str]:
    # template: optional format string using column names e.g. "Human: {instruction}\nAssistant: {response}\n"
    # If None, uses text_column directly.
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)
    texts = []
    scanned = 0
    for example in ds:
        text = template.format(**example) if template else example[text_column]
        scanned += 1
        if len(text) < min_chars:
            continue
        texts.append(text[:200_000])
        if max_examples is not None and len(texts) >= max_examples:
            break
    print(f"  Scanned {scanned}, kept {len(texts)}")
    return texts


def build_dataset(
    tokenizer: BPETokenizer,
    context_length: int,
    sources: list = None,
    personal_dir: str = None,
) -> TextDataset:
    """
    sources: list of dicts, each with keys:
        dataset  - HuggingFace dataset name
        split    - e.g. "train"
        column   - text column name
        max      - max number of examples to load
        min_chars - minimum character length to keep (filters short texts)
    """
    all_texts = []

    for src in (sources or []):
        print(f"\nLoading {src['dataset']} ({src.get('max', 'all')} examples)...")
        all_texts += load_hf_dataset_texts(
            src["dataset"],
            split=src.get("split", "train"),
            text_column=src.get("column", "text"),
            max_examples=src.get("max"),
            min_chars=src.get("min_chars", 0),
            template=src.get("template"),
        )

    if personal_dir:
        all_texts.append(load_personal_data(personal_dir))

    # Encode each text separately — much faster than one giant string
    print(f"\nEncoding {len(all_texts)} texts...", flush=True)
    token_ids = []
    for i, text in enumerate(all_texts):
        print(f"  [{i+1}/{len(all_texts)}] {len(text):,} chars", flush=True)
        token_ids += tokenizer.encode(text)

    return TextDataset(token_ids, context_length)


def get_dataloader(dataset: TextDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

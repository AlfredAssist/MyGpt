import torch
import torch.nn as nn
from torch.utils.data import random_split
from GPT import GPT
from Dataset import build_dataset, get_dataloader
from TokenizationAndBPE import BPETokenizer

VOCAB_SIZE   = 8000   # richer vocabulary for more expressive text
EMBED_DIM    = 192    # bigger representations
N_HEADS      = 4
N_LAYERS     = 6      # deeper = more capable
MAX_SEQ_LEN  = 256    # longer context window
DROPOUT      = 0.1
BATCH_SIZE   = 2      # small batch to avoid OOM segfault on Pi
CONTEXT_LEN  = 256
LR           = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
MAX_STEPS    = 5_000_000
EVAL_EVERY   = 500
DEVICE       = "cpu"

def get_lr(step: int, warmup_steps: int = 200, max_steps: int = MAX_STEPS) -> float:
    # Linear warmup then cosine decay
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

def evaluate(model, dataloader, device, n_batches=20) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= n_batches:
                break
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            total_loss += loss.item()
    model.train()
    return total_loss / min(n_batches, i + 1)

SAVE_EVERY = 2000  # save checkpoint every N steps in case of crash

def save_checkpoint(model, optimizer, step, tok):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": {
            "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "max_seq_len": MAX_SEQ_LEN, "dropout": DROPOUT,
        }
    }, "checkpoint.pt")
    tok.save("tokenizer.json")

def train(model, train_loader, val_loader, optimizer, device, tok, start_step=0):
    import signal

    model.train()
    step = start_step

    def handle_interrupt(sig, frame):
        print(f"\nInterrupt received — saving checkpoint at step {step}...")
        save_checkpoint(model, optimizer, step, tok)
        print("Saved. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    for epoch in range(999):  # loop until max_steps
        for x, y in train_loader:
            if step >= MAX_STEPS:
                print("Training complete.")
                return

            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)                          # [batch, seq, vocab]
            loss   = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),      # [batch*seq, vocab]
                y.view(-1)                             # [batch*seq]
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # Update learning rate
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group['lr'] = lr * LR

            if step % EVAL_EVERY == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"step {step:>5} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | lr {lr*LR:.2e}", flush=True)

            if step % SAVE_EVERY == 0 and step > start_step:
                save_checkpoint(model, optimizer, step, tok)
                print(f"  checkpoint saved at step {step}", flush=True)

            step += 1

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Train tokenizer on 20 novels + 5000 Wikipedia articles for broad vocabulary
    print("Loading tokenizer sample...")
    from datasets import load_dataset
    tok_sample_texts = []

    # Gutenberg novels
    for ex in load_dataset("manu/project_gutenberg", split="en", streaming=True):
        if len(ex["text"]) < 100_000:
            continue
        tok_sample_texts.append(ex["text"][:50_000])
        if len(tok_sample_texts) >= 20:
            break

    # Wikipedia articles
    wiki_count = 0
    for ex in load_dataset("rahular/simple-wikipedia", split="train", streaming=True):
        tok_sample_texts.append(ex["text"][:2_000])
        wiki_count += 1
        if wiki_count >= 5000:
            break

    # News articles
    news_count = 0
    for ex in load_dataset("cc_news", split="train", streaming=True):
        tok_sample_texts.append(ex["text"][:2_000])
        news_count += 1
        if news_count >= 1000:
            break

    # Instruction/response pairs — teaches the Human:/Assistant: pattern
    alpaca_count = 0
    for ex in load_dataset("tatsu-lab/alpaca", split="train", streaming=True):
        tok_sample_texts.append(ex["text"][:2_000])
        alpaca_count += 1
        if alpaca_count >= 2000:
            break

    tok_sample_text = "\n".join(tok_sample_texts)
    tok = BPETokenizer(vocab_size=VOCAB_SIZE)
    tok.train(tok_sample_text)
    print("Tokenizer trained.")

    # 2. Build dataset from multiple sources
    print("Loading dataset...")
    dataset = build_dataset(
        tokenizer=tok,
        context_length=CONTEXT_LEN,
        sources=[
            # Victorian/Edwardian novels — rich formal prose
            {
                "dataset": "manu/project_gutenberg",
                "split": "en",
                "column": "text",
                "max": 150,
                "min_chars": 100_000,
            },
            # Simple Wikipedia — modern clear English on every topic
            {
                "dataset": "rahular/simple-wikipedia",
                "split": "train",
                "column": "text",
                "max": 50_000,
                "min_chars": 200,
            },
            # CC News — 15k modern news articles, diverse real-world topics
            {
                "dataset": "cc_news",
                "split": "train",
                "column": "text",
                "max": 15_000,
                "min_chars": 500,
            },
            # AG News — 120k short news snippets, punchy modern sentences
            {
                "dataset": "ag_news",
                "split": "train",
                "column": "text",
                "max": 120_000,
                "min_chars": 0,
            },
            # Alpaca — 52k instruction/response pairs, teaches question→answer pattern
            {
                "dataset": "tatsu-lab/alpaca",
                "split": "train",
                "column": "text",
                "max": 52_000,
                "min_chars": 0,
            },
            # Dolly — 15k human-written Q&A across categories (factual, creative, etc.)
            {
                "dataset": "databricks/databricks-dolly-15k",
                "split": "train",
                "column": "text",
                "template": "Human: {instruction}\nAssistant: {response}\n",
                "max": 15_000,
                "min_chars": 0,
            },
        ],
    )
    print(f"Dataset size: {len(dataset):,} windows")

    # 3. Split train / val
    val_size   = min(500, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE)
    val_loader   = get_dataloader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # 4. Build model
    model = GPT(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, MAX_SEQ_LEN, DROPOUT).to(DEVICE)
    print(f"Model parameters: {model.get_num_params():,}")

    # 5. Optimizer — exclude biases and LayerNorm from weight decay
    decay_params   = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=LR, betas=(0.9, 0.95))

    # 6. Resume from checkpoint if one exists
    import os
    start_step = 0
    if os.path.exists("checkpoint.pt"):
        print("Resuming from checkpoint.pt...")
        ckpt = torch.load("checkpoint.pt", map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "step" in ckpt:
            start_step = ckpt["step"]
            print(f"Resuming from step {start_step:,} — {MAX_STEPS - start_step:,} steps remaining")

    # 7. Train (saves checkpoint every SAVE_EVERY steps automatically)
    train(model, train_loader, val_loader, optimizer, DEVICE, tok, start_step)

    # 8. Final save
    save_checkpoint(model, optimizer, MAX_STEPS, tok)
    print("Saved checkpoint.pt and tokenizer.json")

import torch
import torch.nn as nn
from torch.utils.data import random_split
from GPT import GPT
from Dataset import build_dataset, get_dataloader
from TokenizationAndBPE import BPETokenizer

# --- Hyperparameters (tuned for Raspberry Pi 5 CPU training) ---
VOCAB_SIZE   = 4000   # smaller vocab = smaller embedding table
EMBED_DIM    = 128    # was 192 — halved to cut params and speed up each step
N_HEADS      = 4      # head_dim = 32, still clean
N_LAYERS     = 4      # was 6 — fewer layers = faster per step
MAX_SEQ_LEN  = 128    # was 256 — shorter context = less memory and faster attention
DROPOUT      = 0.1
BATCH_SIZE   = 4      # small batch for limited RAM
CONTEXT_LEN  = 128
LR           = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
MAX_STEPS    = 5000   # ~2-4 hours on Pi 5, run overnight
EVAL_EVERY   = 200
DEVICE       = "cpu"  # Pi 5 has no GPU

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

def train(model, train_loader, val_loader, optimizer, device):
    model.train()
    step = 0

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
                print(f"step {step:>5} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | lr {lr*LR:.2e}")

            step += 1

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Stream one Gutenberg book and truncate to 100 KB — enough for BPE to learn
    #    the Victorian vocabulary without the O(merges × text_len) cost blowing up.
    print("Loading tokenizer sample from Gutenberg (streaming)...")
    from datasets import load_dataset
    for ex in load_dataset("manu/project_gutenberg", split="en", streaming=True):
        tok_sample_text = ex["text"][:100_000]
        break

    tok = BPETokenizer(vocab_size=VOCAB_SIZE)
    tok.train(tok_sample_text)
    print("Tokenizer trained.")

    # 2. Build dataset — stream 300 Gutenberg books, formal Victorian/Edwardian prose
    print("Loading dataset (streaming)...")
    dataset = build_dataset(
        tokenizer=tok,
        context_length=CONTEXT_LEN,
        hf_dataset_name="manu/project_gutenberg",
        hf_split="en",
        hf_text_column="text",
        hf_max_examples=10,
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

    # 6. Train
    train(model, train_loader, val_loader, optimizer, DEVICE)

    # 7. Save
    torch.save({"model": model.state_dict(), "config": {
        "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "max_seq_len": MAX_SEQ_LEN, "dropout": DROPOUT,
    }}, "checkpoint.pt")
    tok.save("tokenizer.json")
    print("Saved checkpoint.pt and tokenizer.json")

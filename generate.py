import torch
from GPT import GPT
from TokenizationAndBPE import BPETokenizer


def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            ids_cond = ids[:, -model.embed.pos_emb.num_embeddings:]

            logits = model(ids_cond)           # [1, seq, vocab]
            logits = logits[0, -1, :]          # last position: [vocab]

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering — zero out everything outside top k
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = -1e9

            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample
            ids        = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(ids[0].tolist())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/max_lovinger/Documents/LLM')

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load tokenizer
    tok = BPETokenizer(vocab_size=8000)
    tok.load("tokenizer.json")

    # Load model
    checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=False)
    cfg   = checkpoint["config"]
    model = GPT(**cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    prompt = input("Prompt: ")
    output = generate(model, tok, prompt, max_new_tokens=200, temperature=0.8, device=device)
    print(output)

# CLAUDE.md

## What this is
Competition to train the best LM fitting in 16MB compressed artifact.
Metric: val_bpb (bits per byte) on FineWeb val set. Lower is better.
Current SOTA: 1.1428 (10L Int5-MLP + BigramHash(10240), 2026-03-20)
Biggest untapped gains from records: sliding window eval (-0.032), MLP 3x (-0.029), int6 QAT (-0.012)

## Key files
- `train_gpt.py` — main training script AND the competition artifact
- `train_gpt_mlx.py` — local Apple Silicon version for fast iteration
- `records/track_10min_16mb/` — all prior leaderboard submissions with READMEs
- `data/cached_challenge_fineweb.py` — dataset download script

## Run commands

# Local smoke test (Mac, fast iteration)
RUN_ID=smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py

# Remote single H100 (iterating before 8xH100 submission)
RUN_ID=exp1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 train_gpt.py

# Full leaderboard run (8xH100)
RUN_ID=submission1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=8 train_gpt.py

## Constraints (hard rules, never violate)
- Compressed artifact (code + model) must be <16,000,000 bytes
- Training must complete in <10 min on 8xH100 SXM
- No external downloads or network calls during eval
- Cannot access val data during training
- Submissions need 3+ runs showing p<0.01 significance over SOTA

## What to modify
- `train_gpt.py` — architecture, quantization, training loop, eval strategy
- CLAUDE.md — update SOTA score when I beat a record

## What NOT to modify
- `data/` loading logic unless explicitly asked
- Evaluation integrity (BPB calculation, val split)
- `records/` folder contents (read-only reference)

## Known PyTorch Compatibility Issues
- Do NOT use torch.torch_version.TorchVersion() inside forward() or any 
  torch.compile-traced function — dynamo cannot trace it
- Do NOT use enable_gqa= keyword in scaled_dot_product_attention — not
  supported before PyTorch 2.2, and torch.compile traces both branches
  even with init-time booleans. Instead, always manually repeat k/v heads.
- Version checks must be done at __init__ time as a stored boolean attribute,
  never inline inside forward()
- Runpod RTX 4090 pods run an older PyTorch — always write compile-safe code
```

**Claude Code memory** (`~/.claude/CLAUDE.md` — your global one) — good for things true across all your projects, like "always write torch.compile-safe code." But this is competition-specific so CLAUDE.md in the repo is better.

**Custom slash command** — useful if this pattern repeats. Create `.claude/commands/check-compile.md`:
```
Review train_gpt.py for any code that would break torch.compile/dynamo tracing:
- Version checks inside forward()
- Non-traceable Python objects in traced functions
- Any torch._dynamo.exc.Unsupported patterns
Report issues found without fixing them.
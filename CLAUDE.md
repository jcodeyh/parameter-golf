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
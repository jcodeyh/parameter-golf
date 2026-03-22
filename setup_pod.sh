#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y tmux

# Clone and setup repo
cd /workspace
git clone https://github.com/jcodeyh/parameter-golf.git
cd parameter-golf
git checkout exp/layer-looping
pip install -r requirements.txt

# Download dataset (1 shard for iteration, 80 for full runs)
SHARDS=${1:-1}
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards $SHARDS

echo "Setup complete. Run: tmux new -s training"

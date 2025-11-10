cat > README.md << 'EOF'
# markovChains

CLI tools for Markov chains (n-step transitions, stationary distributions) and a random-walk plotter.

## Reproduce
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/markov_chains_cli.py --mode stationary --size 3 --seed 123 --outdir out
EOF


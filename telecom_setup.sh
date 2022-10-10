# don't forget VPN +
ssh nmicaux@gpu6.enst.fr
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  # On Fedora
# pip install requests pandas beautifulsoup4 tqdm wandb transformers datasets scikit-learn  # On Fedora
# pip install requests pandas beautifulsoup4 tqdm wandb torch transformers datasets scikit-learn  # On Ubuntu
python3 -m wandb login
git clone https://github.com/NicolasMICAUX/human_guided_attention.git
cd human_guided_attention/
python human_guided_attention_poc.py
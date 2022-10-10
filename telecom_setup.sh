# don't forget VPN +
ssh nmicaux@gpu2.enst.fr
pip install requests pandas beautifulsoup4 tqdm wandb torch transformers datasets scikit-learn
python3 -m wandb login
git clone https://github.com/NicolasMICAUX/human_guided_attention.git
cd human_guided_attention/
python human_guided_attention_poc.py
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tokenizer_dir = os.path.join(project_root, 'tokenizer')
sys.path.insert(0, tokenizer_dir)

import tokenizer
import torch
from lightning_lm import LMTraining
from config import ModelConfig

tokenizer_pl = tokenizer.PolishToKenizer()
tokenizer_pl.load_vocab(os.path.join(tokenizer_dir, 'vocab.json'))

model_config = ModelConfig()

model_pl = LMTraining.load_from_checkpoint(
    os.path.join(project_root, 'models/PL/last.ckpt'),
    vocab_size=model_config.vocab_size,
    d_model=model_config.d_model,
    num_heads=model_config.num_heads,
    num_layers=model_config.num_layers,
    context_size=model_config.context_size,
    d_ff=model_config.d_ff,
    dropout=model_config.dropout
)

model_pl.eval()

prompts = [
    "Koty są super, ponieważ ",
    "Ten dom stoi na wzgórzu ",
    "Wczoraj wieczorem spotkałem ",
    "Polska jest krajem, który ",
    "Najważniejsze w życiu jest ",
    "Kiedy byłem dzieckiem, zawsze ",
    "Żeby skonstruować bombę, należy ",
    "Kraków znany jest z ",
    "W lesie mieszkał stary ",
    "Recepta na szczęście to ",
    "Nie wierzę, że właśnie ",
    "Gdybym mógł cofnąć czas, ",
    "Moja babcia zawsze mówiła, że ",
    "Na końcu świata znajduje się ",
    "Pewnego dnia wszystko się zmieni, bo ",
    "Sekret dobrego barszczu polega na ",
    "Warszawskie metro jest ",
    "Śnieg padał przez całą noc i ",
    "Kot siedział na parapecie i ",
    "Według naukowców przyszłość ludzkości ",
    "Największym problemem współczesnego świata jest ",
    "O trzeciej w nocy usłyszałem ",
    "Miłość to uczucie, które ",
    "Pierogi ruskie smakują najlepiej z ",
    "Morze Bałtyckie w lato ",
]

for prompt in prompts:
    input_ids_pl = tokenizer_pl.tokens_to_ids(prompt)
    input_tensor_pl = torch.tensor([input_ids_pl], dtype=torch.long)

    with torch.no_grad():
        output_pl = model_pl.model.generate(input_tensor_pl, max_length=40, temperature=0.7)

    result_pl = tokenizer_pl.ids_to_tokens(output_pl[0].tolist())

    print(result_pl)
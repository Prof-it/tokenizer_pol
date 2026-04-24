import torch
from pathlib import Path

from config import ModelConfig
from language_model import LanguageModel
from klej.train_lora import load_tokenizer

CHECKPOINT_PATHS = {
    'pl': Path(__file__).parent.parent / 'models' / 'PL' / 'last.ckpt',
    'spm': Path(__file__).parent.parent / 'models' / 'SPM' / 'last.ckpt',
}

PROMPTS = [
    # Genitive plural — model must produce irregular ending (e.g. -ek, -ów, -y)
    "Na stole leżało kilka list",
    # Dative singular — verb 'pomóc' governs dative, model must inflect the noun
    "Nauczyciel postanowił pomóc swo",
    # Instrumental plural — preposition 'z' forces instrumental (e.g. kolegami, przyjaciółmi)
    "Przez całe życie pracowałem razem z mo",
    # Locative singular — preposition 'o' forces locative (e.g. mężu, Bogu, przyjacielu)
    "Przez cały wieczór rozmawialiśmy o naszym ",
    # Perfective aspect — 'już' forces perfective past, model must pick aspect correctly
    "Kiedy wróciliśmy do domu, nasze dzieci już ",
    # Consonant-alternating stem — e.g. mąż→mężu, Bóg→Bogu, przyjaciel→przyjaciela
    "Opowiedziała mi wszystko o swoim ",
    # Vocative — direct address requires vocative case (e.g. Panie Doktorze, Pani Profesor)
    "Przepraszam, czy mógłbym prosić ",
    # Verbal noun in genitive — 'z powodu' governs genitive of the following noun phrase
    "Spotkanie zostało odwołane z powodu nieoczeki",
]

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9


def load_lm(mode: str, device: torch.device) -> LanguageModel:
    checkpoint_path = CHECKPOINT_PATHS[mode]
    print(f"Loading [{mode}] from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['state_dict']
    # LMTraining wraps LanguageModel under 'model.' prefix
    state_dict = {
        k[len('model.'):]: v
        for k, v in state_dict.items()
        if k.startswith('model.')
    }

    cfg = ModelConfig()
    model = LanguageModel(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        context_size=cfg.context_size,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def generate(model: LanguageModel, tokenizer, prompt: str) -> str:
    ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long)
    output_ids = model.generate(
        input_tensor,
        max_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    # Decode only the newly generated tokens
    new_ids = output_ids[0, len(ids):].tolist()
    return tokenizer.decode(new_ids)


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Device: {device}\n")

    print("Loading models...")
    pl_model = load_lm('pl', device)
    spm_model = load_lm('spm', device)

    print("Loading tokenizers...")
    pl_tok = load_tokenizer('pl')
    spm_tok = load_tokenizer('spm')

    sep = "=" * 70

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{sep}")
        print(f"Prompt {i}: {prompt}")
        print(sep)

        pl_cont = generate(pl_model, pl_tok, prompt)
        spm_cont = generate(spm_model, spm_tok, prompt)

        print(f"[PL]  {prompt}{pl_cont}")
        print()
        print(f"[SPM] {prompt}{spm_cont}")

    print(f"\n{sep}")


if __name__ == '__main__':
    main()

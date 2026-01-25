import sentencepiece as spm
from config import ModelConfig

# 1. train spm on the sama data
spm.SentencePieceTrainer.train(
    input='data/training/tokenizer_data.txt',
    model_prefix='data/spm/baseline_tokenizer',
    vocab_size=ModelConfig.vocab_size,
    model_type='bpe',
    shuffle_input_sentence=True,
    character_coverage=1.0,
    split_by_whitespace=True,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<bos>',
    eos_piece='<eos>',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=['<mask>']
)
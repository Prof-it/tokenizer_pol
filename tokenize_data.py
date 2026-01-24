import sys
sys.path.append('tokenizer/')
import tokenizer
import sentencepiece as spm
import numpy as np
import json

tokenizer_spm = spm.SentencePieceProcessor()
tokenizer_spm.Load('data/spm/baseline_tokenizer.model')

tokenizer_pl = tokenizer.PolishToKenizer()
tokenizer_pl.from_pretrained('tokenizer/vocab.json')

with open('data/training/corpus.jsonl', 'r') as f:
    training_data = [json.loads(line)['text'] for line in f]

def encode_datasets():

    corpus_spm = [tokenizer_spm.Encode(text) for text in training_data]
    np.save('data/model_training/corpus_spm.npy', np.array(corpus_spm, dtype=object))
    print('Saved to data/model_training/corpus_spm.npy')

    corpus_pl = [tokenizer_pl.encode(text) for text in training_data]
    np.save('data/model_training/corpus_pl.npy', np.array(corpus_pl, dtype=object))
    print('Saved to data/model_training/corpus_pl.npy')

    print(f"Tokenized {len(training_data)} samples")
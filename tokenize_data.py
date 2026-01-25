import sys
sys.path.append('tokenizer/')
import tokenizer
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import polars as pl

CORPUS_PATH = 'data/training/corpus.parquet'
OUTPUT_DIR = 'data/model_training/'
SPM_MODEL = 'data/spm/baseline_tokenizer.model'
PL_VOCAB = 'tokenizer/vocab.json'


def tokenize_with_pl(texts, output_path):
    tokenizer_pl = tokenizer.PolishToKenizer()
    tokenizer_pl.from_pretrained(PL_VOCAB)

    output_path = output_path + 'corpus_pl.bin'

    print("\nEncoding with Polish tokenizer...")
    total_tokens = 0
    with open(output_path, 'wb') as f:
        for text in tqdm(texts, desc="PL"):
            chunk = np.array(tokenizer_pl.tokens_to_ids(text), dtype=np.uint16)
            chunk.tofile(f)
            total_tokens += len(chunk)

    print(f"Saved: {output_path} ({total_tokens:,} tokens)")
    return total_tokens


def tokenize_with_spm(texts, output_path):
    tokenizer_spm = spm.SentencePieceProcessor()
    tokenizer_spm.Load(SPM_MODEL)

    output_path = output_path + 'corpus_spm.bin'

    print("\nEncoding with SentencePiece tokenizer...")
    total_tokens = 0
    with open(output_path, 'wb') as f:
        for text in tqdm(texts, desc="SPM"):
            chunk = np.array(tokenizer_spm.EncodeAsIds(text), dtype=np.uint16)
            chunk.tofile(f)
            total_tokens += len(chunk)

    print(f"Saved: {output_path} ({total_tokens:,} tokens)")
    return total_tokens


def main():
    print("Loading corpus...")
    texts = pl.read_parquet(CORPUS_PATH)['text'].to_list()
    print(f"Loaded {len(texts):,} documents")

    total_chars = sum(len(t) for t in texts)

    # tokens_pl = tokenize_with_pl(texts, OUTPUT_DIR)
    # print(f"\nCompression ratio PL:  {total_chars / tokens_pl:.2f} chars/token")

    tokens_spm = tokenize_with_spm(texts, OUTPUT_DIR)
    print(f"Compression ratio SPM: {total_chars / tokens_spm:.2f} chars/token")


if __name__ == '__main__':
    main()
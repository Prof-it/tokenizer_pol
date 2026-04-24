"""
Tokenization metrics comparison — all four tokenizers on the same metrics.

Tokenizers compared
-------------------
  PL Tokenizer    — Polish morphological tokenizer (this repo)
  SPM Baseline    — SentencePiece BPE baseline (this repo)
  Bielik-v3       — speakleash/Bielik-1.5B-v3   (HuggingFace)
  HerBERT         — allegro/herbert-base-cased   (HuggingFace)

Metrics (all tokenizers, same implementation)
---------------------------------------------
1. Fertility Rate        — avg tokens per word (lower = more efficient vocabulary use)
2. STRR                  — Single Token Retention Rate: % of words encoded as 1 token
                           (higher = better vocabulary coverage)
                           Nayeem et al. (2025)
3. Morphological         — For inflected forms (word ≠ lemma), % whose first subword
   Consistency           token matches the lemma's first subword token (after stripping
                         the ▁ / ## boundary marker).  Arnett & Bergen (2025)
4. Morphological         — Pairwise F1 within inflection groups: for every pair of
   Consistency F1        surface forms sharing a lemma, whether they share at least one
                         subword string.  FP = 0 by construction (all pairs share a
                         lemma), so F1 = 2R / (1 + R).  Asgari et al. (2025)
"""

import sys
import re
import random
import json
from collections import defaultdict

sys.path.append('../tokenizer/')
import tokenizer as pl_tok_module

import sentencepiece as spm
import polars as pl
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

CORPUS_PATH = '../data/training/corpus.parquet'
SPM_MODEL   = 'data/spm/baseline_tokenizer.model'
PL_VOCAB    = 'tokenizer/vocab.json'


N_TEXTS_METRICS = 5_000   # texts to draw words from for fertility / STRR
N_WORDS_MAX     = 50_000  # cap on words used for fertility / STRR
N_TEXTS_MORPH   = 500     # texts processed through spaCy
SEED            = 42

SPACE = '\u2581'          # SentencePiece ▁ boundary marker
HASH2 = '##'              # WordPiece continuation marker (HerBERT)
PL_WORD_RE      = re.compile(r'\b[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]{2,}\b')
INTERESTING_POS = {'VERB', 'NOUN', 'ADJ', 'ADV'}

TOKENIZERS_HF = {
    'Bielik-v3 (APT4)':  'speakleash/Bielik-1.5B-v3',
    'HerBERT (CharBPE)': 'allegro/herbert-base-cased',
}

EXAMPLE_WORDS = [
    'pisać', 'napisałem', 'pisze', 'pisałam',
    'dom', 'domów', 'domem', 'domki',
    'szybko', 'szybszy', 'najszybciej',
    'polska', 'polskiego', 'polskim',
]

# Every metric takes a callable  tok_fn(word: str) -> list[str]
# that returns subword *strings* for a single word.  Helpers below create
# these callables for each tokenizer family.

def make_pl_tok_fn(tok_pl):
    return lambda w: tok_pl.tokenize(w)


def make_spm_tok_fn(tok_spm):
    return lambda w: tok_spm.EncodeAsPieces(w)


def make_hf_tok_fn(tokenizer):
    """HF tokenizer → tok_fn returning subword strings without special tokens."""
    return lambda w: tokenizer.convert_ids_to_tokens(
        tokenizer.encode(w, add_special_tokens=False)
    ) or []


def strip_boundary(token: str) -> str:
    return token.lstrip(SPACE).lstrip(HASH2)


# Fertility & STRR

def compute_fertility_strr(words: list[str], tok_fn) -> tuple[float, float]:
    """
    Fertility = avg tokens/word
    STRR      = fraction of words encoded as exactly 1 token
    Words that produce zero tokens are skipped
    """
    total_tokens = single = valid = 0

    for w in tqdm(words, desc='  fertility/STRR', leave=False):
        toks = tok_fn(w)
        n = len(toks)
        if n == 0:
            continue
        valid        += 1
        total_tokens += n
        single       += (n == 1)

    fertility = total_tokens / valid if valid else 0.0
    strr      = single / valid       if valid else 0.0
    return fertility, strr

# Morphological Consistency  (first-token-stem match)

def extract_inflected_pairs(texts: list[str], nlp) -> dict[str, str]:
    pairs: dict[str, str] = {}

    for doc in tqdm(
        nlp.pipe(texts, batch_size=32),
        total=len(texts),
        desc='  spaCy',
        leave=False,
    ):
        for token in doc:
            if (
                token.pos_ in INTERESTING_POS
                and token.is_alpha
                and len(token.text) > 2
                and token.lemma_ not in {'-', '_', ''}
                and token.lemma_.lower() != token.text.lower()
            ):
                pairs[token.text.lower()] = token.lemma_.lower()

    return pairs


def compute_morph_consistency(
    pairs: dict[str, str], tok_fn
) -> tuple[float, int]:
    """
    For each (inflected_form, lemma) pair: are the first subword strings
    (after stripping boundary markers) identical?
    Returns (rate, n_pairs_evaluated).
    """
    consistent = total = 0

    for word, lemma in tqdm(pairs.items(), desc='  morph consistency', leave=False):
        wt = tok_fn(word)
        lt = tok_fn(lemma)
        if not wt or not lt:
            continue
        total += 1
        if strip_boundary(wt[0]) == strip_boundary(lt[0]):
            consistent += 1

    return (consistent / total if total else 0.0), total


# Morphological Consistency F1  (pairwise within inflection groups)

def compute_morph_consistency_f1(
    pairs: dict[str, str], tok_fn
) -> tuple[float, int]:
    """
    Group surface forms by shared lemma; for every pair within a group check
    whether their subword string sets intersect.

    Since all pairs share a lemma by construction, FP = 0 always, and
    F1 = 2·recall / (1 + recall).

    Returns (f1, n_pairwise_comparisons).
    """
    lemma_to_forms: dict[str, list[str]] = defaultdict(list)
    for form, lemma in pairs.items():
        lemma_to_forms[lemma].append(form)

    # only lemmas with ≥ 2 distinct surface forms are informative
    groups = [
        list(set(forms))
        for forms in lemma_to_forms.values()
        if len(set(forms)) >= 2
    ]

    tp = fn = n_pairs = 0

    for forms in tqdm(groups, desc='  morph F1', leave=False):
        tok_sets = {f: set(tok_fn(f)) for f in forms}
        for i, f1 in enumerate(forms):
            for f2 in forms[i + 1:]:
                n_pairs += 1
                if tok_sets[f1] & tok_sets[f2]:
                    tp += 1
                else:
                    fn += 1

    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = 1.0            # FP = 0 by construction
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return f1, n_pairs

def evaluate_tokenizer(
    name: str,
    tok_fn,
    vocab_size: int,
    words: list[str],
    pairs: dict[str, str],
) -> dict:
    print(f'\n  [{name}]')
    fertility, strr = compute_fertility_strr(words, tok_fn)
    mc,    n_mc  = compute_morph_consistency(pairs, tok_fn)
    mc_f1, n_f1  = compute_morph_consistency_f1(pairs, tok_fn)
    return {
        'name':               name,
        'vocab_size':         vocab_size,
        'fertility':          fertility,
        'strr':               strr,
        'morph_consistency':  mc,
        'morph_f1':           mc_f1,
        'n_mc':               n_mc,
        'n_f1':               n_f1,
    }

def print_examples(tok_pl, tok_spm):
    print('\nExample tokenizations:')
    header = f"  {'Word':<20}  {'PL Tokenizer':<40}  SPM"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for word in EXAMPLE_WORDS:
        pl_toks  = tok_pl.tokenize(word)
        spm_toks = tok_spm.EncodeAsPieces(word)
        print(f"  {word:<20}  {str(pl_toks):<40}  {spm_toks}")


def print_results(results: list[dict], n_words: int) -> None:
    W = 94
    print('\n' + '=' * W)
    print(
        f"  {'Tokenizer':<22} {'Vocab':>8}  "
        f"{'Fertility ↓':>11}  {'STRR ↑':>8}  "
        f"{'Morph Cons. ↑':>13}  {'Morph F1 ↑':>11}"
    )
    print('=' * W)
    for r in results:
        print(
            f"  {r['name']:<22} {r['vocab_size']:>8,}  "
            f"{r['fertility']:>11.3f}  {r['strr']*100:>7.1f}%  "
            f"{r['morph_consistency']*100:>12.1f}%  "
            f"{r['morph_f1']*100:>10.1f}%"
        )
    print('=' * W)
    r0 = results[0]
    print(
        f"  Words: {n_words:,}   "
        f"Morph pairs (consistency): {r0['n_mc']:,}   "
        f"Morph pairs (F1): {r0['n_f1']:,}"
    )
    print('=' * W)

def main():
    random.seed(SEED)

    print('Loading corpus ...')
    df = pl.read_parquet(CORPUS_PATH)
    all_texts = df['text'].to_list()

    texts_metrics = random.sample(all_texts, min(N_TEXTS_METRICS, len(all_texts)))
    texts_morph   = random.sample(all_texts, min(N_TEXTS_MORPH,   len(all_texts)))

    all_words = [w for text in texts_metrics for w in PL_WORD_RE.findall(text)]
    words = random.sample(all_words, min(N_WORDS_MAX, len(all_words)))
    print(f'Words sampled for fertility / STRR: {len(words):,}')

    print(f'\nMorphological pairs  ({len(texts_morph)} texts via spaCy)')
    nlp   = spacy.load('pl_core_news_sm')
    pairs = extract_inflected_pairs(texts_morph, nlp)
    print(f'  Unique inflected (word, lemma) pairs found: {len(pairs):,}')

    print('\nLoading local tokenizers ...')
    tok_pl = pl_tok_module.PolishToKenizer()
    tok_pl.load_vocab(PL_VOCAB)

    tok_spm = spm.SentencePieceProcessor()
    tok_spm.Load(SPM_MODEL)

    print('\nEvaluating all tokenizers ...')
    results = []

    results.append(evaluate_tokenizer(
        'PL Tokenizer',
        make_pl_tok_fn(tok_pl),
        len(tok_pl.vocab),
        words, pairs,
    ))
    results.append(evaluate_tokenizer(
        'SPM Baseline',
        make_spm_tok_fn(tok_spm),
        tok_spm.GetPieceSize(),
        words, pairs,
    ))

    for name, hf_id in TOKENIZERS_HF.items():
        print(f'\n  Loading {hf_id} ...')
        hf_tok = AutoTokenizer.from_pretrained(hf_id)
        results.append(evaluate_tokenizer(
            name,
            make_hf_tok_fn(hf_tok),
            hf_tok.vocab_size,
            words, pairs,
        ))

    print_examples(tok_pl, tok_spm)

    print_results(results, len(words))

    out_path = 'tokenizer_eval.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()

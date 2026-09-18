# PLTK: Morphology-Aware BPE Tokenization for Polish Language Models

[![Paper (ACL 2026, in press)](https://img.shields.io/badge/paper-ACL2026-green)](https://github.com/Prof-it/tokenizer_pol) 
[![Presentation (YouTube)](https://img.shields.io/badge/video-KONVENS2026-red)](https://www.youtube.com/watch?v=c0lc5DfHeTE)
[![KONVENS 2026 - Uni Hamburg](https://img.shields.io/badge/conference-KONVENS2026-blue)](https://konvens2026.uni-hamburg.de/index.php/dates-program/)
[![GitHub Repo](https://img.shields.io/badge/repo-github-lightgrey)](https://github.com/Prof-it/tokenizer_pol)

> **PLTK** (PoLish morphological ToKenizer) is an open-source, morphology-aware BPE tokenizer for Polish.  
> It uses part-of-speech-specific trie constraints to enforce morpheme boundaries during subword vocabulary construction.  
> Presented at [KONVENS 2026, Uni-Hamburg](https://konvens2026.uni-hamburg.de/), to appear in ACL 2026.

---

## Table of Contents

- [Overview](#overview)
- [Main Contributions](#main-contributions)
- [Installation](#installation)
- [Quick Start & Usage](#quick-start--usage)
- [Reproducing Results](#reproducing-results)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Links](#links)

---

## Overview

Subword tokenization is crucial for NLP in morphologically rich languages like Polish.  
Standard BPE approaches (e.g., SentencePiece) may fragment inflectional forms and stems.

**PLTK** introduces a novel, POS-aware BPE tokenizer that blocks merges across detected morpheme boundaries using simple trie-based affix constraints.  
This yields more linguistically consistent segmentations, reduces pre-training loss, and improves downstream performance in decoder-only LMs on the Polish KLEJ benchmark.

For a video presentation, see the [KONVENS 2026 talk](https://www.youtube.com/watch?v=c0lc5DfHeTE).

---

## Main Contributions

- 🆕 **PLTK:** Morphology-aware BPE tokenizer for Polish.
- ✔️ Enforces morpheme boundary constraints using POS-specific tries.
- 🏋️ Ablation: GPT-2 models trained from scratch, only difference = tokenizer.
- ⚡ Higher morphological consistency, lower LM pre-training loss, and improved KLEJ scores.
- 📊 Public code, corpus splits, scripts, and evaluation recipes.

---

## Installation

The repository requires Python 3.8+, `spaCy`, and basic deep learning/NLP dependencies.

```bash
git clone https://github.com/Prof-it/tokenizer_pol.git
cd tokenizer_pol
pip install -r requirements.txt
python -m spacy download pl_core_news_sm
# For SentencePiece baseline and model training, install:
pip install sentencepiece torch transformers
```

---

## Quick Start & Usage

### Training a Morphology-Aware PLTK Tokenizer

```bash
# 1. Preprocess your Polish corpus (see paper for details, code/data scripts in ./scripts)
python scripts/preprocess_corpus.py --input raw_corpus.txt --output corpus_cleaned.txt

# 2. Train the PLTK tokenizer (morphology-aware)
python scripts/train_pltk.py --corpus corpus_cleaned.txt --vocab-size 30005 --output pltk_tokenizer.model

# 3. Train the SPM BPE baseline (for ablation)
python scripts/train_spm.py --corpus corpus_cleaned.txt --vocab-size 30005 --output spm_tokenizer.model
```

### Encoding Data and Training Models

```bash
# 4. Encode the full corpus with each tokenizer
python scripts/encode_corpus.py --tokenizer pltk_tokenizer.model --input full_corpus.txt --output pltk_encoded.txt
python scripts/encode_corpus.py --tokenizer spm_tokenizer.model --input full_corpus.txt --output spm_encoded.txt

# 5. Train two GPT-2 models (PLTK vs. SPM), identical hyperparameters
python scripts/train_gpt2.py --data pltk_encoded.txt --tokenizer pltk_tokenizer.model --output-dir model_pltk/
python scripts/train_gpt2.py --data spm_encoded.txt --tokenizer spm_tokenizer.model --output-dir model_spm/
```

### Evaluation

```bash
# 6. Evaluate both models on KLEJ (LoRA fine-tuning for each downstream task)
python scripts/fine_tune_lora.py --model model_pltk/ --task KLEJ_TASK --output results_pltk/
python scripts/fine_tune_lora.py --model model_spm/ --task KLEJ_TASK --output results_spm/
# Scripts for intrinsic metrics: see scripts/metrics.py
```

---

## Reproducing Results

- Corpus: Polish Wikipedia, OPUS-OpenSubtitles, OPUS-ParaCrawl (see paper for filtering/preprocessing).
- Vocabulary size: 30,005.
- Model: GPT-2 base; 12 layers, 12 heads, dim=768 (see full hyperparams in paper).
- Downstream: KLEJ benchmark (NER, entailment, review sentiment, etc.).
- All recipes & key scripts in `scripts/`. See also the docstring/docs in each script for further details.

---

## Results

### Tokenization Quality

| Tokenizer        | FR    | STRR  | MC   | MC-F1 |
|------------------|-------|-------|------|-------|
| **PLTK** (ours)  | 1.608 | 54.2  | 54.9 | 50.7  |
| SPM Baseline     | 1.432 | 69.6  | 33.6 | 16.1  |
| Bielik (APT4)    | 1.438 | 69.4  | 29.9 | 13.3  |
| HerBERT          | 1.387 | 72.8  | 23.3 | 7.7   |

### Downstream (KLEJ benchmark)

| Model | NER   | C-E   | C-R   | IN    | OUT   | AR    | **Avg** |
|-------|-------|-------|-------|-------|-------|-------|---------|
| **PLTK** |0.842|0.922|0.917|0.833|0.605|0.820| **0.823**|
| SPM   |0.839 |0.906 |0.902 |0.817 |0.595 |0.817 |0.813 |

See the full paper for complete methodology, additional qualitative results, and ablation details.

---


## Citation

If you use PLTK in your academic work, please cite:

```bibtex
@inproceedings{adamczykPltkMorphology2026,
  title={PLTK: Morphology-Aware BPE Tokenization for Polish Language Models},
  author={Rafał Adamczyk and Tianxiang Lu and Maja Popovic},
  booktitle={Proceedings of KONVENS 2026},
  year={2026},
  address={Hamburg, Germany},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/venues/konvens/},
  note={Published in the proceedings of KONVENS 2026, to appear in the ACL Anthology}
}
```

> **Note:** The proceedings will be open access and available after the conference at the [ACL Anthology: KONVENS venue](https://aclanthology.org/venues/konvens/).

---

## License

This project is open-sourced under the MIT License.  
See the `LICENSE` file for details.

---

## Contact

- Rafał Adamczyk — [rafal.adamczyk@iu-study.org](mailto:rafal.adamczyk@iu-study.org)
- Tianxiang Lu — [tianxiang.lu@iu.org](mailto:tianxiang.lu@iu.org)
- Maja Popovic — [maja.popovic@iu.org](mailto:maja.popovic@iu.org)

---

## Links

- 📄 [Paper (ACL 2026, TBA)](https://github.com/Prof-it/tokenizer_pol)
- 🖥️ [GitHub Repository](https://github.com/Prof-it/tokenizer_pol)
- 🎥 [YouTube Presentation (KONVENS 2026)](https://www.youtube.com/watch?v=c0lc5DfHeTE)
- 🏛️ [KONVENS 2026, Uni-Hamburg](https://konvens2026.uni-hamburg.de/index.php/dates-program/)

---

For questions, bug reports, or collaboration proposals, please open an issue or contact the authors.

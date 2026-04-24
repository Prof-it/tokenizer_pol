import sentencepiece as spm

SPM_MODEL = 'data/spm/baseline_tokenizer.model'
tokenizer_spm = spm.SentencePieceProcessor()
tokenizer_spm.Load(SPM_MODEL)

print(tokenizer_spm.EncodeAsIds("<pad>"))
print(tokenizer_spm.GetPieceSize())
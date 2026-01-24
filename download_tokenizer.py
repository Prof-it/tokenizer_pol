from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="rafal-adamczyk/polish-morphological-tokenizer",
  local_dir="./tokenizer",
  token=True
)

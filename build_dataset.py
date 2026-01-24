import requests
import os
import gzip
import json
from datasets import load_dataset
import polars as pl

# data config for split
TARGET_ROWS = 250_000
RATIO_OPEN_SUBS = 0.60
RATIO_PARACRAWL = 0.25
RATIO_WIKIPEDIA = 0.15

raw_dir = 'data/raw/'
os.makedirs(raw_dir, exist_ok=True)

training_dir = 'data/training/'
os.makedirs(training_dir, exist_ok=True)


def download_raw_files():
    # 1. Download and save Open Subtitles
    # https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/pl.txt.gz
    txt_filepath = os.path.join(raw_dir, 'open_subtitles_pl.txt')
    gz_filepath = os.path.join(raw_dir, 'open_subtitles_pl.txt.gz')

    if os.path.exists(txt_filepath):
        print(f'Open Subtitles file already exists: {txt_filepath}')
        print(f'File size: {os.path.getsize(txt_filepath) / (1024**3):.2f} GB')
    else:
        print('Downloading Open Subtitles PL...')
        open_subtitles_url = 'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/pl.txt.gz'
        response = requests.get(open_subtitles_url, stream=True)
        response.raise_for_status()

        # Save the gzipped file
        print(f'Saving gzipped file to {gz_filepath}...')
        with open(gz_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Gzipped file saved')

        # Unpack the gzip file
        print(f'Unpacking to {txt_filepath}...')
        with gzip.open(gz_filepath, 'rb') as f_in:
            with open(txt_filepath, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f'Unpacked successfully. File size: {os.path.getsize(txt_filepath) / (1024**3):.2f} GB')

        # Remove the .gz file after unpacking
        print(f'Removing gzipped file: {gz_filepath}')
        os.remove(gz_filepath)
        print('Cleanup complete')

    # 2. Download and save Wikipedia
    # Wikipedia: https://huggingface.co/datasets/chrisociepa/wikipedia-pl-20230401/tree/main
    wiki_filepath = os.path.join(raw_dir, 'wikipedia_pl.jsonl')

    if os.path.exists(wiki_filepath):
        print(f'\nWikipedia file already exists: {wiki_filepath}')
        print(f'File size: {os.path.getsize(wiki_filepath) / (1024**2):.2f} MB')
    else:
        print('\nDownloading Wikipedia PL dataset...')
        wikipedia_data = load_dataset("chrisociepa/wikipedia-pl-20230401")

        # Save Wikipedia dataset as JSON lines
        print(f'Saving Wikipedia data to {wiki_filepath}...')
        with open(wiki_filepath, 'w', encoding='utf-8') as f:
            for split_name, split_data in wikipedia_data.items():
                print(f'Processing split: {split_name} ({len(split_data)} examples)')
                for example in split_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print('Wikipedia dataset saved successfully')

    # 3. Download and save ParaCrawl
    # ParaCrawl: https://object.pouta.csc.fi/OPUS-ParaCrawl/v5/mono/pl.txt.gz
    paracrawl_txt_filepath = os.path.join(raw_dir, 'paracrawl_pl.txt')
    paracrawl_gz_filepath = os.path.join(raw_dir, 'paracrawl_pl.txt.gz')

    if os.path.exists(paracrawl_txt_filepath):
        print(f'\nParaCrawl file already exists: {paracrawl_txt_filepath}')
        print(f'File size: {os.path.getsize(paracrawl_txt_filepath) / (1024**3):.2f} GB')
    else:
        print('\nDownloading ParaCrawl PL...')
        paracrawl_url = 'https://object.pouta.csc.fi/OPUS-ParaCrawl/v5/mono/pl.txt.gz'
        response = requests.get(paracrawl_url, stream=True)
        response.raise_for_status()

        # Save the gzipped file
        print(f'Saving gzipped file to {paracrawl_gz_filepath}...')
        with open(paracrawl_gz_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Gzipped file saved')

        # Unpack the gzip file
        print(f'Unpacking to {paracrawl_txt_filepath}...')
        with gzip.open(paracrawl_gz_filepath, 'rb') as f_in:
            with open(paracrawl_txt_filepath, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f'Unpacked successfully. File size: {os.path.getsize(paracrawl_txt_filepath) / (1024**3):.2f} GB')

        # Remove the .gz file after unpacking
        print(f'Removing gzipped file: {paracrawl_gz_filepath}')
        os.remove(paracrawl_gz_filepath)
        print('Cleanup complete')


def process_raw_files():
    print('Processing raw files')
    # Open Subtitles
    open_subs = pl.scan_csv(
        raw_dir + 'open_subtitles_pl.txt',
        separator='\n',
        has_header=False,
        new_columns=['text'],
        ignore_errors=True,
        quote_char=None
    ).with_columns(
        # Chain operations: remove dashes, then quotes
        pl.col('text')
            .str.strip_chars_start('- ')
            .str.strip_chars_start('-')
            .str.strip_chars()
            .str.replace_all('"', '')  # Remove all double quotes
            .alias('text')
    ).with_columns(pl.lit('open_subs').alias('source'))

    # ParaCrawl
    paracrawl = pl.scan_csv(
        raw_dir + 'paracrawl_pl.txt',
        separator='\n',
        has_header=False,
        new_columns=['text'],
        ignore_errors=True,
        quote_char=None
    ).with_columns(
        pl.col('text').str.replace_all('"', '').alias('text')  # Remove all double quotes
    ).with_columns(pl.lit('paracrawl').alias('source'))

    # Wikipedia
    wiki = pl.scan_ndjson(raw_dir + 'wikipedia_pl.jsonl').select(
        pl.col('text')
    ).with_columns(
        # Strip leading/trailing whitespace and newlines
        pl.col('text').str.strip_chars()
        # Replace all newlines and multiple whitespace with single space
        .str.replace_all(r'\s+', ' ')
        # Remove 'Przypisy' or 'Bibliografia' to the end
        .str.replace(r'\s*(Przypisy|Bibliografia).*$', '')
        # Final strip to clean up
        .str.strip_chars()
    ).with_columns(pl.lit('wikipedia').alias('source'))

    # concat into one lf
    big_df = pl.concat([open_subs, paracrawl, wiki])

    # filter out examples with less than 50 chars
    big_df = big_df.filter(pl.col('text').str.len_chars() >= 50).filter(pl.col('text').is_not_null() & (pl.col('text') != ''))


    pol_pattern = r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9\s\.,!?;:()\[\]\-"\']'

    big_df = big_df.with_columns(
        pl.col('text')
        .str.replace_all('"', '')
        .str.replace_all(pol_pattern, '')
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
    )


    # collect to sample
    full_df = big_df.collect()
    print(f'Total rows: {len(full_df):,}')


    # Sample from each source
    open_subs_sample = full_df.filter(pl.col('source') == 'open_subs').sample(
        n=int(TARGET_ROWS * RATIO_OPEN_SUBS),
        shuffle=True,
        seed=42
    )

    paracrawl_sample = full_df.filter(pl.col('source') == 'paracrawl').sample(
        n=int(TARGET_ROWS * RATIO_PARACRAWL),
        shuffle=True,
        seed=42
    )

    wikipedia_sample = full_df.filter(pl.col('source') == 'wikipedia').sample(
        n=int(TARGET_ROWS * RATIO_WIKIPEDIA),
        shuffle=True,
        seed=42
    )

    # Combine and shuffle
    tokenizer_df = pl.concat([open_subs_sample, paracrawl_sample, wikipedia_sample]).sample(
        fraction=1.0,
        shuffle=True,
        seed=42
    )

    # SAVE TOKENIZER DATA (plain text - one line per document)
    tokenizer_path = os.path.join(training_dir, 'tokenizer_data.txt')
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        for text in tokenizer_df['text']:
            # Ensure no newlines in text (newlines = document boundaries)
            cleaned = ' '.join(text.split())
            f.write(cleaned + '\n')

    # SAVE FULL CORPUS (JSONL - for LLM training)
    corpus_path = os.path.join(training_dir, 'corpus.jsonl')
    full_df.select('text').write_ndjson(corpus_path)


    print(f'\nOutputs:')
    print(f'  1. Tokenizer dataset: {tokenizer_path}')
    print(f'     - Rows: {len(tokenizer_df):,}')
    print(f'     - Size: {os.path.getsize(tokenizer_path) / (1024**2):.2f} MB')
    print(f'  2. Full corpus: {corpus_path}')
    print(f'     - Rows: {len(full_df):,}')
    print(f'     - Size: {os.path.getsize(corpus_path) / (1024**3):.2f} GB')


def data_pipeline():
    download_raw_files()
    process_raw_files()
    print('\nDone.')

data_pipeline()
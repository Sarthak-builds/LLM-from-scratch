"""

--- What is SentencePiece? ---
  SentencePiece (Google, 2018) is a language-independent tokenizer that:
    - Treats input as a raw stream of Unicode characters (no pre-tokenization needed).
    - Supports both BPE and Unigram LM algorithms.
    - Trains its OWN vocabulary directly from your corpus.
    - Used in: T5, ALBERT, LLaMA, Mistral, mT5, and many multilingual models.

  Key difference from tiktoken:
    - tiktoken uses a FIXED pre-trained vocabulary (OpenAI's).
    - SentencePiece TRAINS a new vocabulary from scratch on YOUR data.

  Under the hood (BPE mode):
    - Whitespace is treated as a special character '▁' (U+2581).
    - This lets it be fully reversible without needing the original whitespace info.
"""

import os
import re
import sentencepiece as spm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "assets", "llm-book.txt")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("=" * 60)
print("  BPE TOKENIZER — SentencePiece (trained on llm-book.txt)")
print("=" * 60)
print(f"Total characters in book: {len(raw_text)}")
print(f"First 99 characters: {repr(raw_text[:99])}\n")

model_prefix = os.path.join(base_dir, "assets", "llm_book_spm")
model_file   = model_prefix + ".model"

if not os.path.exists(model_file):
    print("Training SentencePiece BPE model on llm-book.txt ...")
    spm.SentencePieceTrainer.train(
        input=file_path,
        model_prefix=model_prefix,
        vocab_size=8000,        # target vocabulary size
        model_type="bpe",       # algorithm: BPE
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,              
        eos_id=3,              
    )
    print("Model trained and saved!\n")
else:
    print("Pre-trained model found, loading ...\n")


sp = spm.SentencePieceProcessor()
sp.load(model_file)

print(f"Vocabulary size: {sp.get_piece_size()}")

token_ids = sp.encode(raw_text)
print(f"Total tokens after encoding: {len(token_ids)}")

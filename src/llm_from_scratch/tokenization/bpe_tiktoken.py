import tiktoken
from .._paths import LLM_BOOK_PATH

raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")

print(f"Total characters in book: {len(raw_text)}")
print(f"First 99 characters: {repr(raw_text[:99])}\n")

enc = tiktoken.get_encoding("gpt2")

print(f"Vocabulary size (gpt2): {enc.n_vocab}")

token_ids = enc.encode(raw_text, allowed_special="all")
print(f"Total tokens after encoding: {len(token_ids)}")



import re
from .._paths import LLM_BOOK_PATH


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

if __name__ == "__main__":
    raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")

    print("Total number of characters:", len(raw_text))
    
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("Total number of tokens:", len(preprocessed))
    all_words = sorted(list(set(preprocessed)))
    
    # Add special tokens
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    
    vocab_size = len(all_words)
    vocab = {token:integer for integer,token in enumerate(all_words)}
    print("Vocabulary size:", vocab_size)
    
    tokenizer = SimpleTokenizerV2(vocab)


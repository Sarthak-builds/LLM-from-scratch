import re
import os

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
    # Pointing to the llm-book.txt 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "assets", "llm-book.txt")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
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
    
    print("\n--- Running Tokenizer V2 Example ---")

    example_text = "This is a completely unknownwordto test the <|unk|> functionality <|endoftext|>"
    print(f"Original Text: '{example_text}'")
    encoded_ids = tokenizer.encode(example_text)
    print(f"Encoded IDs:   {encoded_ids}")
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded Text:  '{decoded_text}'")

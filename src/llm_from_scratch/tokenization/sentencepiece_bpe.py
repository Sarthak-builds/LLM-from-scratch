import sentencepiece as spm
from .._paths import LLM_BOOK_PATH, SPM_MODEL_PREFIX

raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")

model_file = SPM_MODEL_PREFIX.with_suffix(".model")

if not model_file.exists():
    print("Training SentencePiece BPE model on llm-book.txt ...")
    spm.SentencePieceTrainer.train(
        input=str(LLM_BOOK_PATH),
        model_prefix=str(SPM_MODEL_PREFIX),
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
sp.load(str(model_file))

print(f"Vocabulary size: {sp.get_piece_size()}")

token_ids = sp.encode(raw_text)
print(f"Total tokens after encoding: {len(token_ids)}")
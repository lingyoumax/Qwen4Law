from tokenizers import ByteLevelBPETokenizer
import os

tokenizer = ByteLevelBPETokenizer()

files=["中国民法典.txt"]

tokenizer.train(files=files, vocab_size=500, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])

save_dir = "BLBPE_tokenizer"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

tokenizer.save_model(save_dir)

if __name__ == "__main__":
    tokens = tokenizer.encode("财产")
    print(tokens.tokens)
    print(tokens.ids)

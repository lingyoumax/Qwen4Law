from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from modelscope import PreTrainedTokenizerFast, AutoTokenizer

from tools import getFiles

# 加载 Qwen3 tokenizer（用于拿 special tokens）
tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
save_dir = "Qwen3-Embedding-0.6B"
tk.save_pretrained(save_dir)

# 创建 Byte-Level BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

# 不提前加 special tokens，先正常训练
trainer = trainers.BpeTrainer(
    vocab_size=tk.vocab_size,   # 这里可以根据需要调小，比如 30k，更常见
    min_frequency=2,
)

directory = 'laws'
fileend = '.txt'

files = [f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)]
tokenizer.train(files, trainer)

# 包装成 PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer
)

# 后追加 special tokens（保持 Qwen3 的逻辑）
hf_tokenizer.add_special_tokens({
    "eos_token": tk.special_tokens_map['eos_token'],
    "pad_token": tk.special_tokens_map['pad_token'],
    "additional_special_tokens": tk.special_tokens_map['additional_special_tokens']
})

save_dir = "BLBPE_tokenizer"
hf_tokenizer.save_pretrained(save_dir)

# 重新加载验证
new_tokenizer = AutoTokenizer.from_pretrained("BLBPE_tokenizer")

print("原 tokenizer vocab_size:", tk.vocab_size)
print("新 tokenizer vocab_size:", new_tokenizer.vocab_size)

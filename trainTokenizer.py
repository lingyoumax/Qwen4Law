from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from modelscope import PreTrainedTokenizerFast, AutoTokenizer

from tools import getFiles

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
tokenizer = Tokenizer(models.BPE())
split_pre_tokenizer = pre_tokenizers.Split(
    pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    behavior="isolated",
)
bytelevel_pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False,
    trim_offsets=False,
    use_regex=False,
)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([split_pre_tokenizer, bytelevel_pre_tokenizer])
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)

tokenizer.normalizer = normalizers.NFC()

trainer = trainers.BpeTrainer(
    vocab_size=tk.vocab_size,
    min_frequency=2,
)

directory='laws'
fileend='.txt'

tokenizer.train(files=[f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)], trainer=trainer)
#tokenizer.add_special_tokens(tk.all_special_tokens)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    add_prefix_space=False,
    add_bos_token=False
)

hf_tokenizer.add_special_tokens({
    "eos_token": tk.special_tokens_map['eos_token'],
    "pad_token": tk.special_tokens_map['pad_token'],
    "additional_special_tokens": tk.special_tokens_map['additional_special_tokens']
})

hf_tokenizer.unk_token = None
hf_tokenizer.bos_token = None

# 其他属性
hf_tokenizer.clean_up_tokenization_spaces = False
hf_tokenizer.model_max_length = 131072
hf_tokenizer.errors = "replace"
hf_tokenizer.split_special_tokens = False

save_dir = "Tokenizer"
hf_tokenizer.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained("Tokenizer")

print(tk.vocab_size)

print(tokenizer.vocab_size)
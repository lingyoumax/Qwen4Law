from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors
from modelscope import PreTrainedTokenizerFast, AutoTokenizer

from tools import getFiles
save_dir = "Tokenizer"
tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
tokenizer = Tokenizer(models.BPE(continuing_subword_prefix="",end_of_word_suffix=""))
split_pre_tokenizer = pre_tokenizers.Split(
    pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    behavior="isolated",
)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    split_pre_tokenizer,
    pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
])

trainer = trainers.BpeTrainer(
    vocab_size=tk.vocab_size,
    min_frequency=2,
)

directory='laws'
fileend='.txt'

tokenizer.train(files=[f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)], trainer=trainer)
#tokenizer.add_special_tokens(tk.all_special_tokens)
tokenizer.model.save(save_dir)

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

hf_tokenizer._tokenizer.decoder=decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
hf_tokenizer._tokenizer.normalizer = normalizers.NFC()

hf_tokenizer._tokenizer.post_processor = processors.Sequence([
    processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A $B <|endoftext|>",
        special_tokens=[
            ("<|endoftext|>", hf_tokenizer.convert_tokens_to_ids("<|endoftext|>"))
        ]
    )
])

hf_tokenizer.unk_token = None
hf_tokenizer.bos_token = None

hf_tokenizer.clean_up_tokenization_spaces = False
hf_tokenizer.model_max_length = 131072
hf_tokenizer.errors = "replace"
hf_tokenizer.split_special_tokens = False


hf_tokenizer.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(save_dir)

print(tk.vocab_size)

print(tokenizer.vocab_size)
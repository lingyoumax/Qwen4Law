from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Tokenizer', padding_side='left')

# 原始字符串
text = "中国"
print(tokenizer.tokenize("中国"))


# 编码为 token IDs（默认返回 dict）
encoded = tokenizer(text)

print("input_ids:", encoded['input_ids'])
print("token_type_ids:", encoded.get('token_type_ids'))
print("attention_mask:", encoded['attention_mask'])
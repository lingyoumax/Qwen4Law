from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Tokenizer', padding_side='left')

text = "中国"

encoded = tokenizer(text)

print("input_ids:", encoded['input_ids'])
print("token_type_ids:", encoded.get('token_type_ids'))
print("attention_mask:", encoded['attention_mask'])
print("tokens",tokenizer.convert_ids_to_tokens(encoded['input_ids']))
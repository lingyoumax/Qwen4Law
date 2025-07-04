from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer("BLBPE_tokenizer/vocab.json", "BLBPE_tokenizer/merges.txt")

texts = [
    "这是一个测试",
    "中华人民共和国民法典",
    "自然人享有民事权利能力"
]

total_tokens = 0

for text in texts:
    encoded = tokenizer.encode(text)
    total_tokens += len(encoded.tokens)
    print(f"句子: {text}\nToken数: {len(encoded.tokens)}\nTokens: {encoded.tokens}\n")

avg_tokens = total_tokens / len(texts)
print(f"✅ 平均 token 数: {avg_tokens:.2f}")

text = "中华人民共和国"
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.tokens}")
print(f"被拆成了 {len(encoded.tokens)} 个子词")

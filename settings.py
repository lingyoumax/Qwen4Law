import torch

retriever_modelname = "Qwen/Qwen3-Embedding-0.6B"
reranker_modelname = "Qwen/Qwen3-Reranker-0.6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 7
num_negative_docs = 10
embedding_max_length = 512
embedding_batch_size = 4
embedding_test_ratio = 0.2
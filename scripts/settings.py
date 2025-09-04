import torch

embedding_modelname = "Qwen/Qwen3-Embedding-0.6B"
reranker_modelname = "Qwen/Qwen3-Reranker-0.6B"
llm_modelname = "Qwen/Qwen3-8B"
rewardmodel_modelname = "Qwen/Qwen3-0.6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 7
num_negative_docs = 10
embedding_max_length = 512
embedding_batch_size = 4
embedding_test_ratio = 0.2
reranker_max_length = 512
reranker_batch_size = 4
reranker_test_ratio = 0.2
llm_max_length = 1024
llm_batch_size=4
llm_test_ratio = 0.2
rewardmodel_max_length = 1024
rewardmodel_batch_size=4
rewardmodel_test_ratio = 0.2
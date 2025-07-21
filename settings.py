import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed=7
num_negative_docs=10
max_length = 512
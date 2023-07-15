import torch
from tokenizer import find_token_position
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, texts, queries, answers, model_pretrained = "bert-base-uncased"):
        self.encodings = find_token_position(texts, queries, answers, model_pretrained)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


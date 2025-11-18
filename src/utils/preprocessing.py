import json

import numpy as np
import torch
from torch.utils.data import Dataset


def load_embeddings(file_path):
	embeddings = []
	with open(file_path, "r") as f:
		for line in f:
			nums = np.fromstring(line.strip(), sep=" ")
			n_tokens = int(nums[0])
			dim = int(nums[1])
			vecs = nums[2:].reshape(n_tokens, dim)
			embeddings.append(torch.tensor(vecs, dtype=torch.float32))
	return embeddings


def load_labels(file_path):
	labels = []
	with open(file_path, "r") as f:
		for line in f:
			parts = line.strip().split()
			val = int(parts[0])
			labels.append(val)
	return labels


def load_sentences(file_path):
	result = []
	with open(file_path, "r") as f:
		for line in f:
			parts = line.split("	")
			result.append({"sentences1": parts[0], "sentences2": parts[1], "labels": int(parts[2])})
	return result


class SentencePairDataset(Dataset):
	def __init__(self, emb1_list, emb2_list, labels):
		self.emb1_list = emb1_list
		self.emb2_list = emb2_list
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return {"emb1": self.emb1_list[idx], "emb2": self.emb2_list[idx], "label": self.labels[idx]}


def collate_fn(batch):
	s1 = [instance["emb1"] for instance in batch]
	s2 = [instance["emb2"] for instance in batch]
	labs = [instance["label"] for instance in batch]

	dim = s1[0].size(1)
	maxlen1 = max(x.size(0) for x in s1)
	maxlen2 = max(x.size(0) for x in s2)

	def pad(sentences, maxlen):
		padded = torch.zeros(len(sentences), maxlen, dim)
		for i, s in enumerate(sentences):
			padded[i, : s.size(0)] = s
		return padded

	return pad(s1, maxlen1), pad(s2, maxlen2), torch.tensor(labs, dtype=torch.float32)


class SentencePairDatasetWithEmbeddings(Dataset):
	def __init__(self, emb1_list, emb2_list, sen1_list, sen2_list, labels):
		self.emb1_list = emb1_list
		self.emb2_list = emb2_list
		self.sen1_list = sen1_list
		self.sen2_list = sen2_list
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return {
			"emb1": self.emb1_list[idx],
			"emb2": self.emb2_list[idx],
			"sen1": self.sen1_list[idx],
			"sen2": self.sen2_list[idx],
			"label": self.labels[idx],
		}


def collate_fn_with_embeddings(batch):
	e1 = [instance["emb1"] for instance in batch]
	e2 = [instance["emb2"] for instance in batch]
	s1 = [instance["sen1"] for instance in batch]
	s2 = [instance["sen2"] for instance in batch]
	labs = [instance["label"] for instance in batch]

	dim = e1[0].size(1)
	maxlen1 = max(x.size(0) for x in e1)
	maxlen2 = max(x.size(0) for x in e2)

	def pad(sentences, maxlen):
		padded = torch.zeros(len(sentences), maxlen, dim)
		for i, s in enumerate(sentences):
			padded[i, : s.size(0)] = s
		return padded

	return pad(e1, maxlen1), pad(e2, maxlen2), torch.stack(s1), torch.stack(s2), torch.tensor(labs, dtype=torch.float32)


if __name__ == "__main__":
	data = load_sentences("../TechaArena_Poland_2025/data/TechArena_FormalDataset_EN_TRAIN.dat")

	with open("../datasets/sentences.json", "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)



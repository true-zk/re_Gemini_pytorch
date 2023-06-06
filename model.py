import torch
import torch.nn as nn
import torch.nn.functional as F


# Gemini_feature_size: 9  # 整理后为9


class EmbedLayer(torch.nn.Module):
	def __init__(self, embedding_depth, embedding_size):
		super(EmbedLayer, self).__init__()
		self.embedding_depth = embedding_depth
		self.embedding_size = embedding_size
		self.P_list: list = []
		for i in range(self.embedding_depth):
			self.P_list.append(nn.Parameter(torch.randn(self.embedding_size, self.embedding_size)))

	def forward(self, input_):
		"""
		:param input_: shape = batch * embedding_size * nodes
		:return: Pn * ... Relu(P3 * Relu(P2 * Relu(P1 * Relu(P0 * input_))))
		"""
		curr_embedding = torch.einsum('ik,akj->aij', self.P_list[0], input_)
		curr_embedding = F.relu(curr_embedding)

		for i in range(1, self.embedding_depth - 1):
			curr_embedding = torch.einsum('ik,akj->aij', self.P_list[i], curr_embedding)
			curr_embedding = F.relu(curr_embedding)

		curr_embedding = torch.einsum('ik,akj->aij', self.P_list[self.embedding_depth - 1], curr_embedding)

		return curr_embedding

	def return_embedding_depth(self):
		return self.embedding_depth


def compute_graph_embedding(adj_mat: torch.tensor, feature_mat: torch.tensor, W1, W2, Iter, embed_layer):
	"""
	:param adj_mat: batch * max_nodes * max_nodes
	:param feature_mat: batch * max_nodes * 9
	:param W1: embedding_size * 9
	:param W2: embedding_size * embedding_size
	:param Iter: embedding Iter
	:param embed_layer: class EmbedLayer(torch.nn.Module)
	:return: graph_embedding  # embedding_size * batch
	"""
	feature_mat = torch.einsum('aij->aji', feature_mat)  # batch * 9 * max_nodes

	curr_embedding = torch.zeros((adj_mat.shape()[1], embed_layer.embedding_size))  # max_nodes * embedding_size
	prev_embedding = torch.einsum('aik,kj->aij', adj_mat, curr_embedding)  # batch * max_nodes * embedding_size
	prev_embedding = torch.einsum('aij->aji', prev_embedding)  # batch * embedding_size * max_nodes
	for i in range(Iter):
		neighbor_embedding = embed_layer(prev_embedding)  # batch * embedding_size * max_nodes
		term = torch.einsum('ik,akj->aij', W1, feature_mat)  # batch * embedding_size * max_nodes
		curr_embedding = F.tanh(term + neighbor_embedding)  # batch * embedding_size * max_nodes

		prev_embedding = curr_embedding
		prev_embedding = torch.einsum('aij->aji', prev_embedding)  # batch * max_nodes * embedding_size
		prev_embedding = torch.einsum('aik,akj->aij', adj_mat, prev_embedding)  # batch * max_nodes * embedding_size
		prev_embedding = torch.einsum('aij->aji', prev_embedding)  # batch * embedding_size * max_nodes
	graph_embedding = torch.sum(curr_embedding, dim=2)  # batch * embedding_size
	graph_embedding = torch.einsum('ij->ji', graph_embedding)  # embedding_size * batch
	graph_embedding = torch.matmul(W2, graph_embedding)  # embedding_size * batch
	graph_embedding = torch.einsum('ij->ji', graph_embedding)  # batch * embedding_size
	return graph_embedding


class Gemimi(nn.Module):
	def __init__(self, embedding_size, Gemini_feature_size, embedding_depth, Iter: int):
		"""
		:param embedding_size: embedding_size
		:param Gemini_feature_size: Gemini_feature_size actually 9
		:param embedding_depth: num of P
		:param Iter: embedding Iter times
		"""
		super(Gemimi, self).__init__()
		self.embed_layer = EmbedLayer(embedding_depth, embedding_size)
		self.W1 = nn.Parameter(torch.rand((embedding_size, Gemini_feature_size), dtype=torch.float32) * 0.1)
		self.W2 = nn.Parameter(torch.rand((embedding_size, embedding_size), dtype=torch.float32) * 0.2)
		self.Iter = Iter
		self.cosine_loss = nn.CosineEmbeddingLoss()

	def forward(self, g0, g1, y):
		g0_adj_mat, g0_feature_mat = g0
		g1_adj_mat, g1_feature_mat = g1
		g0_embedding = compute_graph_embedding(g0_adj_mat, g0_feature_mat, self.W1, self.W2, self.Iter, self.embed_layer)
		g1_embedding = compute_graph_embedding(g1_adj_mat, g1_feature_mat, self.W1, self.W2, self.Iter, self.embed_layer)
		# batch * embedding_size
		sim = F.cosine_similarity(g0_embedding, g1_embedding, dim=0)  # batch
		loss = self.cosine_loss(g0_embedding, g1_embedding, y)
		return loss, sim

	def cfg2vec(self, g):
		g_adj_mat, g_feature_mat = g
		g_embedding = compute_graph_embedding(g_adj_mat, g_feature_mat, self.W1, self.W2, self.Iter, self.embed_layer)
		return g_embedding

import sys
import pickle
import numpy as np
from torch.utils.data import IterableDataset
from Genius_Gemini_data.raw_graphs import raw_graphs, raw_graph  # include networkx and os
from Genius_Gemini_data.raw_graphs import *

version = ["openssl-101f"]
arch = ["x86"]
compiler = ["gcc", "clang"]
optimizer = ["O0", "O1", "O2", "O3"]
cfg_dir_name = r'data\extracted-acfg-openssl'
train_set = r'data\train.pk'
val_set = r"data\val.pk"

max_nodes_threshold = 500
min_nodes_threshold = 0
Buffer_Size = 1000
mini_batch = 10


class MyDataset(IterableDataset):
	def __init__(self, file_path, max_nodes=max_nodes_threshold):
		super(MyDataset).__init__()
		self.max_nodes = max_nodes
		self.file_path = file_path
		with open(file_path, "rb") as f:
			self.func_dict = pickle.load(f)

		self.func_name_list = list(self.func_dict.keys())  # func_name is key
		self.func_name_length = len(self.func_name_list)

	# no use
	def __getitem__(self, index):
		pass

	def __iter__(self):
		for func_name in self.func_name_list:
			# per same name func
			same_func_list = self.func_dict[func_name]
			if len(same_func_list) < 2:
				continue
			for i, graph in enumerate(same_func_list):
				g_adj_mat = self.zero_padded_adjmat(graph)
				g_feature_mat = self.feature_vector(graph)
				for flag in range(2):
					if flag == 0:
						# positive sample
						g1_index = np.random.randint(low=0, high=len(same_func_list))
						while g1_index == i:  # not be identical
							g1_index = np.random.randint(low=0, high=len(same_func_list))
						graph1 = same_func_list[g1_index]
						g1_adj_mat = self.zero_padded_adjmat(graph1)
						g1_feature_mat = self.feature_vector(graph1)
						pair = ([g_adj_mat, g_feature_mat], [g1_adj_mat, g1_feature_mat], 1)
						yield pair
					else:
						# negative sample
						name_index = np.random.randint(low=0, high=self.func_name_length)
						while self.func_name_list[name_index] == func_name:  # not be same name
							name_index = np.random.randint(low=0, high=self.func_name_length)
						g2_index = np.random.randint(low=0, high=len(self.func_dict[self.func_name_list[name_index]]))
						graph2 = self.func_dict[self.func_name_list[name_index]][g2_index]
						g2_adj_mat = self.zero_padded_adjmat(graph2)
						g2_feature_mat = self.feature_vector(graph2)
						pair = ([g_adj_mat, g_feature_mat], [g2_adj_mat, g2_feature_mat], -1)
						yield pair

	def zero_padded_adjmat(self, graph):

		def adjmat(g):
			return nx.adjacency_matrix(g).toarray().astype('float32')

		size = self.max_nodes
		unpadded = adjmat(graph)
		padded = np.zeros((size, size))
		if len(graph) > size:
			padded = unpadded[0:size, 0:size]
		else:
			padded[0:unpadded.shape[0], 0:unpadded.shape[1]] = unpadded
		return padded

	def feature_vector(self, graph):
		size = self.max_nodes
		feature_mat = np.zeros((size, 9))
		for _node in graph.nodes:
			if _node == size:
				break
			feature = np.zeros((1, 9))
			vector = graph.nodes[_node]['v']
			num_const = vector[0]
			if len(num_const) == 1:
				feature[0, 0] = num_const[0]
			elif len(num_const) >= 2:
				feature[0, 0:2] = np.sort(num_const)[::-1][:2]
			feature[0, 2] = len(vector[1])
			feature[0, 3:] = vector[2:]
			feature_mat[_node, :] = feature
		return feature_mat


class Str2Byte:
	def __init__(self, file):
		self.file = file

	def read(self, size):
		return self.file.read(size).encode()

	def readline(self, size=-1):
		return self.file.readline(size).encode()


def read_cfg():
	func_dict = {}
	counts = []
	for arch_ in arch:
		count = 0
		for version_ in version:
			for compiler_ in compiler:
				for op_level in optimizer:
					file_name = '_'.join([version_, arch_, compiler_, op_level, "openssl"])
					file_name = file_name + ".cfg"
					file_path = os.path.join(cfg_dir_name, file_name)
					with open(file_path, 'r') as f:
						pickle_ = pickle.load(Str2Byte(f))

					for func in pickle_.raw_graph_list:  # perfectly fit the extractor output
						if len(func.g) < min_nodes_threshold:
							continue
						if func_dict.get(func.funcname) is None:
							func_dict[func.funcname] = []
						func_dict[func.funcname].append(func.g)
						count += 1
		counts.append(count)
	print("for three arch: ", counts)
	return func_dict


def dataset_split_save(func_dict: dict):
	func_name_num = len(func_dict)
	train_num = int(func_name_num * 0.8)
	val_num = func_name_num - train_num

	train_name_index = np.random.choice(list(func_dict.keys()), size=train_num, replace=False)
	train_func_dict = {}
	for func_name in train_name_index:
		train_func_dict[func_name] = func_dict[func_name]
		func_dict.pop(func_name)

	with open(train_set, "wb") as f:
		pickle.dump(train_func_dict, f)

	val_func_dict = func_dict
	with open(val_set, "wb") as f:
		pickle.dump(val_func_dict, f)

	print(f"train set length: {train_num}, val set length: {val_num}")


if __name__ == "__main__":
	# sys.path.append(r"D:\Desktop\Gemini_re\Genius_Gemini_data\raw_graphs.py")
	dataset_split_save(read_cfg())
	# for file in os.listdir(r'data\extracted-acfg-openssl'):
	# 	file_path = os.path.join(r'data\extracted-acfg-openssl', file)
	# 	with open(file_path, 'r') as f:
	# 		tmp = Str2Byte(f)
	# 		# tmp = raw_graphs(tmp)
	# 		pickle_ = pickle.load(tmp)
	# 		tmp = pickle_
	# 		print(type(tmp))
	# 		for g in tmp.raw_graph_list:
	# 			graph = g.g
	# 			print(g)
	# 			for node_ in graph.nodes:
	# 				print(node_)
	# 				vec = g.g.nodes[node_]['v']
	# 				print(vec)
	# 	break

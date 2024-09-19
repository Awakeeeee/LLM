import torch
import re
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
	#maxL是滑动窗口的尺寸
	#stride是遍历间隔,如1就是每隔一个token滑动一下窗口
	def __init__(self, txt, tokenizer, maxL, stride):
		self.input_ts = [] #将会是tensor的列表
		self.predict_ts = []
		tokens = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
		for i in range(0, len(tokens) - maxL, stride):
			window = tokens[i : i + maxL]
			move = tokens[i + 1 : i + maxL + 1]
			self.input_ts.append(torch.tensor(window))
			self.predict_ts.append(torch.tensor(move))
			#dataset的数据结构[tensor([...]), tensor([...])]

	def __len__(self):
		return len(self.input_ts)

	def __getitem__(self, idx):
		return self.input_ts[idx], self.predict_ts[idx]


def CreateDataLoader(txt, batch = 4, maxL = 256, stride = 128, shuffle = True, drop = True, workers = 0):
	tokenizer = tiktoken.get_encoding("gpt2")
	dataset = GPTDataset(txt, tokenizer, maxL, stride)
	loader = DataLoader(dataset, batch_size = batch, shuffle = shuffle, drop_last = drop, num_workers = workers)
	return loader



if __name__ == "__main__":
	
	#现在训练模型的人类语言预测能力
	#这次将故事文本作为希望他会预测的学习样本使用,而不是用于构建vacab
	with open('theverdict.txt', 'r', encoding='utf-8') as f:
		rawtext = f.read()

	#loader = CreateDataLoader(rawtext, batch = 8, maxL = 4, stride = 4, shuffle = False)
	tokenizer = tiktoken.get_encoding("gpt2")
	dataset = GPTDataset(rawtext, tokenizer, 4, 4)
	loader = DataLoader(dataset, batch_size = 8, shuffle = False, drop_last = True, num_workers = 0)

	#仅为学习使用
	torch.manual_seed(123)
	
	vocab_size = tokenizer.n_vocab #gpt2词汇库容量50257
	vector_dim = 256
	#对比来看Embedding和Linear的用法类似,应该都是神经网络中处理不同任务的层
	#embedding是嵌入层,用于把tokenID这种离散数据映射为n维连续向量
	#训练的目的是不断修正参数,让相关词的向量相近,和不相关词的向量远离
	#Embedding类的两个传参是为了构造vocab行+vector列的向量矩阵,最终矩阵的每一行就是一个tokenID的向量
	embedding_layer = torch.nn.Embedding(vocab_size, vector_dim)
	#print(embedding_layer.weight)

	#所以关键是nn.Embedding是怎么构造这个矩阵的,有了矩阵,所谓embedding就是挑出某一行而已

	#embedding layer和data loader的关系是:loader遍历的batch作为layer查询的输入参数
	#layer要一个任意维度（应该是）的tensor作为输出,并按照该tensor的结构把每个数字id替换为vector输出

	it = iter(loader)
	read, predict = next(it)
	#拿第一个batch做实验
	v = embedding_layer(read)
	print(v.shape) #[8,4,256] 8和4是保持输入tensor维度,256是里面的每个数字id都替换成了256维向量

	#尺寸为4的词汇表是因为loader窗口的尺寸为4,只需要表示1234四个位置
	pos_layer = torch.nn.Embedding(4, vector_dim)
	#为1234各生成对应的256维vector,该vector将会加到tokenID的vector上以表示这个ID的顺序
	v_pos = pos_layer(torch.arange(4))
	print(v_pos.shape)

	#最终得到了loader第一个batch对应的embedding,切其中含有位置信息
	batch_eb = v + v_pos
	print(batch_eb.shape)










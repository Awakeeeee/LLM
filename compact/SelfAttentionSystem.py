import torch
import torch.nn as nn

class SASv1(nn.Module): #nn.Module是神经网络层基类
	
	def __init__(self, din, dout):
		super().__init__()
		self.dout = dout
		self.wquery = nn.Parameter(torch.rand(din, dout)) #rand不是随机 是创建(in, out)形状的张量
		self.wkey = nn.Parameter(torch.rand(din, dout))
		self.wval = nn.Parameter(torch.rand(din, dout))

	def forward(self, x): #forward类似神经网络执行入口 x是输入的文本vectors
		keys = x @ self.wkey 		#用来和query点乘计算权重的张量
		queries = x @ self.wquery	#query指的是这个张量而不是原始输入
		vals = x @ self.wval		#用来和权重运算得到最终context
		scores = queries @ keys.T
		factor = self.dout
		weights = torch.softmax(scores / factor ** 0.5, dim=-1)
		context = weights @ vals
		return context


class SASv2(nn.Module):
	
	def __init__(self, din, dout, qkv_bias=False):
		super().__init__()
		self.dout = dout
		self.wquery = nn.Linear(din, dout, qkv_bias)
		self.wkey = nn.Linear(din, dout, qkv_bias)
		self.wval = nn.Linear(din, dout, qkv_bias)

	def forward(self, x):
		#Linear参数的qkv
		keys = self.wkey(x)
		queries = self.wquery(x)
		vals = self.wval(x)
		scores = queries @ keys.T
		factor = self.dout
		weights = torch.softmax(scores / factor ** 0.5, dim=-1)
		context = weights @ vals
		return context

class SAS_causal(nn.Module):
	def __init__(self, din, dout, qkv_bias=False):
		super().__init__()
		self.dout = dout
		self.wquery = nn.Linear(din, dout, qkv_bias)
		self.wkey = nn.Linear(din, dout, qkv_bias)
		self.wval = nn.Linear(din, dout, qkv_bias)

	def get_masked_weights(self, L, weights):
		#这是在softmax之后做causal处理的方法
		#构造一个全1矩阵并只取下半边(tril)用于修改weights
		unit = torch.ones(L, L)
		mask = torch.tril(unit)
		#星号是简单的元素对应相乘,@才是矩阵乘法
		mask_weight = weights * mask

		#然后重新归一化
		#方法是每行元素都除以该行的元素之和
		#dim参数:对于二维张量来说,1就是按行求和,0是按列求和,不填这个参数就是全加起来
		#keepdim参数:结果tensor的shape保持不变
		rowsum = mask_weight.sum(dim=1, keepdim=True)
		mask_weight_norm = mask_weight / rowsum
		print(mask_weight_norm)
		return mask_weight_norm

	def get_masked_softmax(self, scores):
		#这是在softmax之前做causal处理的方法
		#softmax -> e^xi / (e^x1 + e^x2 + ... + e^xn), for 1 <= i <= n
		#按照softmax的具体算法,如果给出的x值为负无穷,那么e^x就是0,所以交给softmax的矩阵可以是半边为-inf的(半边是0就不行,因为e^0=1而不是0)
		L = scores.shape[0]
		unit = torch.ones(L, L)
		half = torch.triu(unit, diagonal=1) #取上三角
		bool_mask = half.bool() #将矩阵的值转为bool,0是False,这样做是为了使用masked_fill函数
		mask = scores.masked_fill(bool_mask, -torch.inf) #masked_fill是按掩码替换的函数,对应True的地方替换为参数指定的值
		#print(mask)
		return mask

	def forward(self, x):
		keys = self.wkey(x)
		queries = self.wquery(x)
		vals = self.wval(x)
		scores = queries @ keys.T
		factor = self.dout

		#weights = torch.softmax(scores / factor ** 0.5, dim=-1)
		#如果weights矩阵对角线一侧都归0 其实就实现了causal attention

		mask = self.get_masked_softmax(scores)
		weights = torch.softmax(mask / factor ** 0.5, dim=-1)
		print(weights)

		context = weights @ vals
		return context
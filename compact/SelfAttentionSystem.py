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


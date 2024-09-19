import torch
#print(torch.__version__)
#print(torch.cuda.is_available())
import torch.nn.functional as MathFunc
from torch.autograd import grad

t0 = torch.tensor(23)
t1 = torch.tensor([3.5, 5.6, 7])
t2 = torch.tensor([[2, 2], [4, 4]])
t3 = torch.tensor([[[1, 2, 1], [2, 2, 1]], [[3, 5, 3], [5, 2, 6]]])

print("check data type of a tensor: ", t3.dtype)

print("check shape(each dimention length) of tensor:", t3.shape)

print("reshape: ", t3.view(3, 2, 2))

print("transpose(a type of reshape): ", t3.mT)

m1 = torch.tensor([[2, 3], [4, 5], [1, 6]]) #3 x 2
m2 = torch.tensor([[3, 4, 5], [6, 7, 8]]) #2 x 3

print("tensor way of matrix multiplication: ", m1 @ m2)

# y = torch.tensor([1.0]) #即使只是一个数字,也用一维tensor来表示
# x1 = torch.tensor([1.1])
# w1 = torch.tensor([2.2], requires_grad=True)
# b = torch.tensor([0.0], requires_grad=True)
# z = x1 * w1 + b
# a = torch.sigmoid(z)
# loss = MathFunc.binary_cross_entropy(a, y)
# loss.backward() #跑一遍backwards propagation,结果是能得到各个参数的梯度
# print(w1.grad)
# print(b.grad)

class NeuralNetwork(torch.nn.Module):
	def __init__(self, input_num, output_num):
		super().__init__()

		#sequential的作用是将下面的layer依次执行
		self.layers = torch.nn.Sequential(
			#1
			torch.nn.Linear(input_num, 30), #input个输入维度,30个输出维度的层
			torch.nn.ReLU(), #设定激活函数是relu

			#2
			torch.nn.Linear(30, 20), #输入和1层的输出维度一样
			torch.nn.ReLU(),

			#output
			torch.nn.Linear(20, output_num)
		)

	#forward含义是从输入层到输出层运行神经网络
	def forward(self, x): #x是输入数据列表(tensor形式的列表,也就是tensor([[a, b, c...]]))
		logits = self.layers(x)
		return logits


m = NeuralNetwork(50, 3)
print(m)
print(sum(p.numel() for p in m.parameters() if p.requires_grad))

print(m.layers[0].weight) # y = xWT + b中的W矩阵(tensor表示)

inputs = torch.rand((1, 50)) #随机生成50个数模拟输入数据
with torch.no_grad():
	out = m(inputs) #nn.Module重载了__call__,实际上是调用forward(x)函数
	print(out)

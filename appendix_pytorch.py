import torch
#print(torch.__version__)
#print(torch.cuda.is_available())

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
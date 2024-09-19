import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ToyDataset(Dataset):
	def __init__(self, features, labels):
		self.features = features
		self.labels = labels

	def __getitem__(self, idx):
		f = self.features[idx]
		l = self.labels[idx]
		return f,l

	def __len__(self):
		return self.labels.shape[0]



features_1 = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, 1.1], [2.7, -1.5]]) #例如,每个元素是邮件的2种特征
labels_1 = torch.tensor([0, 0, 0, 1, 1]) #例如,前三个是普通邮件,后两个是垃圾邮件

features_2 = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
labels_2 = torch.tensor([0, 1])

train = ToyDataset(features_1, labels_1)
test = ToyDataset(features_2, labels_2)

print(len(train))

torch.manual_seed(123)
#batch_size 遍历数据按批次进行,这个参数表示以几个元素合一批
#shuffle 元素会进行随机排序,这是算法有意做的
train_loader = DataLoader(dataset = train, batch_size = 2, shuffle = True, num_workers = 0)
test_loader = DataLoader(dataset = test, batch_size = 2, shuffle = False, num_workers = 0)

for i, (x,y) in enumerate(train_loader):
	print(f"Batch {i}:", x, " -- ", y)
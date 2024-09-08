import re

class Tokenizer:

	def __init__(self, vocab):
		self.forward = vocab # [string,id] dictionary
		self.reverse = {i:s for s,i in vocab.items()} # i2s

	#将输入文本转为token ids
	def encode(self, text):
		preprocessed = re.split(r'([,.()?!"_\']|--|\s)', text)
		preprocessed = [s.strip() for s in preprocessed if s.strip()]
		ids = [self.forward[s] for s in preprocessed]
		return ids

	#输入id列表返回对应完整的文本
	def decode(self, ids):
		mapping = [self.reverse[i] for i in ids]
		text = " ".join(mapping) #join函数把字符串列表合并成一个字符串并用指定字符做连接(这里用空格连接形成句子)
		#因为token scheme一般会把标点符号单独切分,所以目前join的结果可能类似it ' s good,我们还需要把多余的空格给去掉
		#re.sub基本用法sub('匹配', '替换为', 文本)
		#r'\s+([...])'匹配一个或多个空格接上[]内指定符号的形式,也就是那些多余的空格处
		#r'\1'的意思是"对第一个捕获组的引用",捕获组指的是regex表达式中的括号()
		text = re.sub(r'\s+([.,?!()"\'])', r'\1', text)
		return text


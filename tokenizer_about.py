import torch
import re
from SimpleTokenizer import Tokenizer
import tiktoken

if __name__ == '__main__':

	with open('theverdict.txt', 'r', encoding='utf-8') as f:
		rawtext = f.read()

	#print("raw text length: ", len(rawtext))
	#print("preview:\n", rawtext[:99])


	#taste = "Let's firstly do...a simple test."
	#tastsplit = re.split(r'([,.:;?_!"()\']|--|\s)', taste) #r的作用相当于C#字符串前的的@, [,.]匹配[]中的每个字符, \s表示匹配空格,被匹配到的字符会单独切分出来
	#tastefinal = [c for c in tastsplit if c.strip()] #有点像Linq的快捷语法, strip()返回去除空格后的字符,而python中''在if判断等于false
	#print(tastefinal)


	preprocess = re.split(r'([,.?_!"()\']|--|\s)', rawtext)
	preprocess = [item.strip() for item in preprocess if item.strip()]
	print("preprocessed token amount: ", len(preprocess))
	print("preprocessed token preview: ", preprocess[:30])

	tokens_set = sorted(list(set(preprocess))) #转set用于去重,但set是无序结构,所以转回list排序
	tokens_set.extend(["<|endoftext|>", "<|unk|>"])
	#tokens_set.extend("BOS") #begining of sequence
	#tokens_set.extend("EOS")
	#tokens_set.extend("PAD") #padding 用于batch text填充短文本以便让所有文本有统一长度
	vocab_len = len(tokens_set)
	print("vocabulary size: ", vocab_len)
	
	vocabulary = {token:i for i,token in enumerate(tokens_set)} #enumerate用于在遍历容器的同时给出索引值(i, element)
	# print("vocalulary preview:")
	# for i,t in enumerate(vocabulary.items()):
	# 	if i > 50:
	# 		break
	# 	print(t)

	#注意,这时得到的vocalbulary类似一个训练后的模型
	#也就是说虽然他是由这篇具体的小文章训练出来的
	#但是本质上他已经可以用来给任意原始文本做token ID映射, 只是随着规模变大他的"知识"会不够用


	teststr = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the notwordno."
	#----通过简单Tokenizer解释原理
	#ter = Tokenizer(vocabulary)
	#ints = ter.encode(teststr)
	#print(ter.decode(ints))


	#真正的Tokenizer过于复杂,直接用tiktoken库
	#tiktoken库在本地安装有固有的词库,不需要传入某个给予小故事搞出来的词汇表
	realter = tiktoken.get_encoding("gpt2")
	ints = realter.encode(teststr, allowed_special={"<|endoftext|>"})
	print(ints)
	print(realter.decode(ints))

	noword = "Akwirw ier"
	ints = realter.encode(noword) #一般来说应该转为2个数字token,不过因为输入是个没有记载的文本,所以BPE用已知的其他token组合该文本,导致结果是多于2个token
	print(ints)
	for i in ints:
		print(realter.decode([i]))
	print(realter.decode(ints))

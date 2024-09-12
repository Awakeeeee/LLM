import torch
import re
import tiktoken

if __name__ == "__main__":
	
	#现在训练模型的人类语言预测能力
	#这次将故事文本作为希望他会预测的学习样本使用,而不是用于构建vacab
	with open('theverdict.txt', 'r', encoding='utf-8') as f:
		rawtext = f.read()

	ter = tiktoken.get_encoding("gpt2")
	original_encode = ter.encode(rawtext)
	print(len(original_encode))
	tokens = original_encode[50:] #? 不要前50个token 说会比较奇怪


	#???TODO这个参数很重要,他的直观意义是模型能看到多少个token的上下文进行预测,这个意义是后面算法方式决定的
	context_size = 4
	x = tokens[:context_size] #取index 0~(context_size-1)的子数组
	y = tokens[1:context_size+1] #取index 1~context_size的子数组
	#也就是y总是比x向前推进一个token,直观上看这就是逐个单词的预测过程
	print(f"x:{x}")
	print(f"y:	{y}")

	for i in range(1, context_size+1):
		context = tokens[:i] #i excluded
		predict = tokens[i] #i included without :
		print(context, "---->", predict)
		print(ter.decode(context), "---->", ter.decode([predict]))
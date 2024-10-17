import torch
from compact.SelfAttentionSystem import SASv1, SASv2, SAS_causal

#测试用输入,应该是由将文本通过nn.Embedding转出来
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

def single_atten_context_example():
  print(inputs.shape) #6,3 分别是输入的token数和embedding向量维度

  query = inputs[1] #一般当前要为哪个元素计算权重表,这个元素就叫query
  w2 = torch.empty(inputs.shape[0]) #初始化空tensor表示attention score,对于每个输入token都计算一个值,所以是个token数量一样维度的张量
  for i,token in enumerate(inputs):
  	w2[i] = torch.dot(token, query)
  print(w2)

  #目前m2中各项是点乘结果,正负不定且可能大于1,现在做化归到0~1
  w2 = torch.softmax(w2, dim = 0)
  print(w2)

  #最终相关性tensor是个维度和embedding一样的张量
  context2 = torch.zeros(query.shape)
  for i, word in enumerate(inputs):
    mod = w2[i] * word
    context2 += mod
  print(context2)


#single_atten_context_example()

def calculate_all_atten_context():
  #6_1=每个词的权重分数向量,6_2=每个向量含6个元素分别对应每个其他词
  #或理解成类似unity中physics matrix的n*n矩阵关系
  attn_scores = torch.empty(6, 6)

  # for i, query in enumerate(inputs):
  #   for j, other in enumerate(inputs):
  #     score = torch.dot(other, query)
  #     attn_scores[i,j] = score
  # print("attention score tensor: ", attn_scores)

  #用matrix multiplication代替以上O(n^2) for loop
  attn_scores = inputs @ inputs.T
  print("attention score tensor: ", attn_scores)
  #注意这次dim=1,dim参数的意思是,在前面张量参数的第几个维度上进行计算
  attn_weight = torch.softmax(attn_scores, dim=1)
  print("attention weight tensor: ", attn_weight)
  #[6,6] @ [6,3] = [6,3]类似矩阵乘法规则, 看来矩阵乘法可以代替各种for loop
  attn_context = attn_weight @ inputs
  print(attn_context)

#calculate_all_atten_context()


def single_scaled_dot_product_attention():
  x2 = inputs[1] #只算第二个词的attention context
  
  #这个算法需要用到3个中间量
  torch.manual_seed(123)
  d_in = inputs.shape[1]
  d_out = 2
  #3x2的可训练矩阵参数,nn.Parameter用于生成可训练的参数,当传入2维张量时生成的就是可训练的矩阵
  #注:这里的w代表的weight是神经网络概念下的参数权重,不要和self-attention概念下的相关性权重混淆
  w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
  w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
  w_val = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

  #与simple相比,query是这个计算结果,而不直接就是input
  query2 = x2 @ w_query #[1,3]x[3,2] = [1,2]
  keys = inputs @ w_key #[6,3]x[3,2] = 6个[1,2]
  vals = inputs @ w_val

  #与simple相比,权重分数不是两个单词embedding直接dot,而是代表两者的中间量相乘
  #TODO 原理不明白,大概是为了后续训练一直调w矩阵,毕竟不能直接调输入的embedding
  scores_2 = query2 @ keys.T #[1,2]x[2,6] = [1,6]

  #TODO 原理还是不明白,这次的normalization和中间量维度挂钩(除以维度的平方根),说是训练时排除纬度数变大时带来的影响
  dim_factor = keys.shape[-1]
  print(dim_factor)
  #注:softmax传入dim=-1指在前面tensor参数的最后一个维度上进行计算
  weights_2 = torch.softmax(scores_2 / dim_factor ** 0.5, dim=-1)
  print(weights_2)

  #和simple的区别是,汇总时不是直接和input相乘,而是用values代表输入值
  context_2 = weights_2 @ vals #[1,6]x[6,2] = [1,2]
  print(context_2)


#整体感受是相比simple,scaled_dot为输入添加了一个可训练参数,当计算时不直接用输入进行,而是用封装后的中间量进行,该中间量既由输入得出,又带有可训练参数供后续调节
#single_scaled_dot_product_attention()

# torch.manual_seed(789)
# s1 = SASv1(inputs.shape[1], 2)
# c1 = s1(inputs)
# print("v1:", c1)
# torch.manual_seed(789)
# s2 = SASv2(inputs.shape[1], 2)
# c2 = s2(inputs)
# print("v2:", c2)

#关于书中v1 v2如何得到相同结果的实验
#要让结果一样,就是让Linear和Parameter构造的可训练权重矩阵一样,但矩阵ctor中是随机生成的(这也是他们一开始结果不同的原因),所以考虑把一方的复制给另一方
#拷贝前,需要搞清两个问题:
#1.Linear和Parameter虽然都构造了可训练权重矩阵,区别是Linear内部的存储做了转置
#2.拷贝什么字段? Linear构造后的权重储存为weight,该字段包含矩阵张量+斜率信息,通过weight.data只取其中张量部分;而Parameter则直接通过data获取张量
# s1.wquery.data = s2.wquery.weight.data.T
# s1.wkey.data = s2.wkey.weight.data.T
# s1.wval.data = s2.wval.weight.data.T
# c3 = s1(inputs)
# print(c3) #same as c2


torch.manual_seed(789)
s = SAS_causal(inputs.shape[1], 2)
c = s(inputs)
print(c)

#context咋用
import torch

# weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
# print(weights_named)

img_t = torch.randn(3, 5, 5) # shape [channels, rows,. columsn]
weights = torch.tensor([0.2126, 0.7152, 0.0722])
batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columsn]
# print('img_t: ', img_t)
# print('batch_t: ', batch_t)

weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
# weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['columns'])
print('weights_named: ', weights_named.shape, weights_named.names)

# 有一个张量并且想要为其添加名称但不改变现有的名称时，
# 我们可以对其调用refine_names()方法
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named  = batch_t.refine_names(..., 'channels', 'rows', 'columns')

print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

#align_as()方法返回一个张量，其中添加了缺失的维度，现有的维度按正确的顺序排列如下：
weights_aligned = weights_named.align_as(img_named)
print('weight_aligned: ', weights_aligned.shape, weights_aligned.names)


# sum
gray_named = (img_named * weights_aligned)
print('gray_named: ', gray_named.shape, gray_named.names)
gray_named = gray_named.sum('channels')
print('gray_named: ', gray_named.shape, gray_named.names)



print("img[..., :3] named:", img_named[..., :3].shape, img_named[..., :3].names)

gray_named = (img_named[..., : 3] * weights_aligned)
print('gray_named: ', gray_named.shape, gray_named.names)

# error
# gray_named = (img_named[..., : 3] * weights_named)


# 恢复未命名变量
gray_plain = gray_named.rename(None)
print('gray_plain: ', gray_plain.shape, gray_plain.names)

print('-------------未命名方法--------------')


# In[2]:
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])
print('img_t: ', img_t.shape)
print('weights: ', weights.shape)

# In[3]:
batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]
print('batch_t: ', batch_t.shape)


# In[4]
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
print('img_gray_naive: ', img_gray_naive.shape)
print('batch_gray_naive: ', batch_gray_naive.shape)

# In[5]
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
print('unsqueezed_weight: ', unsqueezed_weights.shape)
img_weights = (img_t*unsqueezed_weights)
print('img_weights: ', img_weights.shape)
batch_weights = (batch_t*unsqueezed_weights)
print('batch_weights: ', batch_weights.shape)

img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
print('img_gray_weighted: ', img_gray_weighted.shape)
print('batch_gray_weighted: ', batch_gray_weighted.shape)

# test
a = torch.ones(3)
a[0] = 1
a[1] = 2
a[2] = 3
b = a.unsqueeze(1)
c = b.unsqueeze(0)
print(a)
print(a.shape)
print(b)
print(b.shape)
print(c)
print(c.shape)



















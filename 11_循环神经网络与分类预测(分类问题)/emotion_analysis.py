import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(),
                  ["NN", "V", "DET", "NN"])]

word_to_idx = {}
tag_to_idx = {}
for context, tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)
idx_2_tag={tag_to_idx[i]:i for i in tag_to_idx}

alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i

def make_sequence(x, dic): # 字符编码
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx


class char_lstm(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(char_lstm, self).__init__()

        self.char_embed = nn.Embedding(n_char, char_dim)
        #具有升维功能，每个序列用一个向量表示(seq,batch)->(seq,batch,in_feature=char_dim)
        self.lstm = nn.LSTM(char_dim, char_hidden)

    def forward(self, x):
        x = self.char_embed(x)
        out, _ = self.lstm(x)
        return out[-1,:,:]  # 取出输出序列的最后一个


class lstm_tagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, word_dim,
                 char_hidden, word_hidden, n_tag):
        super(lstm_tagger, self).__init__()
        self.word_embed = nn.Embedding(n_word, word_dim)
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.word_lstm = nn.LSTM(word_dim + char_hidden, word_hidden)
        self.classify = nn.Linear(word_hidden, n_tag)

    def forward(self, x, word):
        char = []
        for w in word:  # 对于每个单词做字符的 lstm
            char_list = make_sequence(w, char_to_idx)
            char_list = char_list.unsqueeze(1)  # (seq, batch, feature) 满足 lstm 输入条件 feature在词嵌入时进行升维操作
            char_infor = self.char_lstm(Variable(char_list))  # (batch, char_hidden)
            char.append(char_infor)

        char = torch.stack(char, dim=0)  # 原本char中的每个元素都是二维变量(batch, feature)，通过此函数在0维处升维至(seq, batch, feature)
        #其实这个地方x，可以在之前用unsqueeze(1)，就不用进行换维
        x = self.word_embed(x)  # (batch, seq, word_dim)
        x = x.permute(1, 0, 2)  # 改变顺序 换维操作
        x = torch.cat((x, char), dim=2)  # 沿着特征通道将每个词的词嵌入和字符 lstm 输出的结果拼接在一起
        x, _ = self.word_lstm(x)

        s, b, h = x.shape
        x = x.view(-1, h)  # 重新 reshape 进行分类线性层
        out = self.classify(x)
        print('out:',out.size())
        return out

net = lstm_tagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_sequence(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        tag = make_sequence(tag, tag_to_idx)
        word_list = Variable(word_list)
        tag = Variable(tag)
        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.data
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))

test_sent = 'Everybody ate the apple'
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())
prediction=torch.max(out,1)[1]
predicted_tag=[idx_2_tag[prediction.numpy()[i]] for i in range(prediction.size(0))]
print(predicted_tag)
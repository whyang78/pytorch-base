1、采用RNN对mnist数据集进行分类预测，mnist在load进来后，维度是[b,c,w,h],其中c=1。   
    采用rnn时，是将一幅图片的h作为seq,w作为feature,即seq=28,input_size=28。同时   
    注意RNN的输入dim=3，所以要降维处理，squeeze一下则将c去掉，数据量没有变化。
2、情感分析这个要结合词向量，需要有NLP方面的知识，不予详细解释，lstm.ipynb则是使用   
    torch.text和Grove来做的。另一个则是自己定义向量，处理方法比较有趣，对每个词语   
    分别进行charlstm（对象是字符），然后对整个句子再进行wordlstm（对象是单词),最后   
    拼接特征向量进行最后的运算。
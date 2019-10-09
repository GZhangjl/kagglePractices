import pandas as pd
import collections
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn, data as gdata, loss as gloss
from mxnet.contrib import text
import os

dir_path = './jigsaw-toxic-comment-classification-challenge/'

_, test_file, test_label_file, train_file = [os.path.join(dir_path, file_name) \
                                             for file_name in sorted(os.listdir(dir_path))]

train_data_labels = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
test_label = pd.read_csv(test_label_file)

# labels_name = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'N']
# labels_index = list(range(1, 8))

# train_data_labels = pd.concat((train_data_labels, \
#                     pd.DataFrame(data=np.zeros((len(train_data_labels), 1)), columns=['N'])), axis=1)
# train_data_labels['N'][train_data_labels[labels_name[:-1]].apply(sum, axis=1) == 0] = 1

labels_name = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
labels_index = list(range(1, 7))

train_data = train_data_labels[['id', 'comment_text']]
train_test_data = pd.concat((train_data, test_data), axis=0)  # 将训练数据与预测数据进行拼接一起形成单词表

# 分词函数


def get_tok(data):
    def _tok(one):
        return [w.lower() for w in one.strip().replace('\n', ' ').split(' ')]
    return [_tok(s) for s in data]

# 形成单词表函数


def get_voc(data):
    word_count = collections.Counter([w for s in get_tok(data) for w in s])
    return text.vocab.Vocabulary(counter=word_count)

# 形成可以用来创建词向量的单词索引


def get_features(data, voc):
    max_len = 100
    
    def padding(v):
        return v[:max_len] if len(v) > max_len else v + [0] * (max_len - len(v))
    return [padding(voc.to_indices(s)) for s in get_tok(data)]


voc = get_voc(train_test_data['comment_text'])

train_features = get_features(train_data['comment_text'], voc)
test_features = get_features(test_data['comment_text'], voc)

# 形成训练数据集的labels变量

in_vars = locals()
for i in labels_index[:-1]:
    in_vars['train_label_{0}'.format(i)] = train_data_labels[labels_name[i-1]]  # 类别 N 不用管

# 形成训练数据集

batch_size = 200

for i in labels_index:
    in_vars['train_set_{0}'.format(i)] = gdata.ArrayDataset(train_features, in_vars['train_label_{0}'.format(i)])
    in_vars['train_iter_{0}'.format(i)] = gdata.DataLoader(in_vars['train_set_{0}'.format(i)], batch_size, shuffle=True)

# 开始设计神经网络模型。本问题中，虽然看似六分类问题，实质上是正对每一个分类的二分类问题。现就针对每一个分类按照二分类问题进行网络构建。
# 考虑参考textCNN思路构建网络


class TextCNN(nn.HybridBlock):
    def __init__(self, voc, w2v_size, k_sizes, n_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(voc), w2v_size)
        self.convs = nn.Sequential()
        for k, c in zip(k_sizes, n_channels):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))
        self.pooling = nn.GlobalMaxPool1D()
        self.dropout_1 = nn.Dropout(0.3)
        self.dense_h = nn.Dense(6, activation='relu')
        self.dropout_2 = nn.Dropout(0.6)
        self.dense = nn.Dense(2)
        
    def hybrid_forward(self, F, inputs, *args, **kwargs):
        emb = self.embedding(inputs)
        input_emb = emb.transpose((0, 2, 1))
        conv_process = [nd.flatten(self.pooling(conv(input_emb))) for conv in self.convs]
        dense_inputs = nd.concat(*conv_process, dim=1)
        hid = self.dense_h(self.dropout_1(dense_inputs))
        outputs = self.dense(self.dropout_2(hid))
        return outputs


w2v_size, k_sizes, n_channels = 10, [3, 4, 5], [10, 10, 10]
lr, wd, num_epochs = 0.005, 0, 5

for i in labels_index:
    in_vars['net_{0}'.format(i)] = TextCNN(voc, w2v_size=w2v_size, k_sizes=k_sizes, n_channels=n_channels)
    net = in_vars['net_{0}'.format(i)]
    net.initialize(init.Xavier())
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    loss = gloss.SoftmaxCrossEntropyLoss()

    labels = in_vars['train_label_{0}'.format(i)]
    train_iter = in_vars['train_iter_{0}'.format(i)]
    train_l_sum, train_acc_sum, n = 0., 0., 0.
    count = 0
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
            print(count)
            count += 1
    print('train_iter_{0}  loss {1}, train_acc {2}'.format(i, train_l_sum / n, train_acc_sum / n))

for i in labels_index:
    filename = './models_params_file/model_params_{0}'.format(labels_name[i-1])
    net = in_vars['net_{0}'.format(i)]
    net.save_parameters(filename)

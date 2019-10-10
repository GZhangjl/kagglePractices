import pandas as pd
import collections
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet.contrib import text
import os

dir_path = './jigsaw-toxic-comment-classification-challenge/'

_, test_file, test_label_file, train_file = [os.path.join(dir_path, file_name) \
                                             for file_name in sorted(os.listdir(dir_path))]

train_data_labels = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
test_label = pd.read_csv(test_label_file)


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

test_features = get_features(test_data['comment_text'], voc)


class TextCNN(nn.HybridBlock):
    def __init__(self, voc, w2v_size, k_sizes, n_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(voc), w2v_size)
        self.convs = nn.HybridSequential()
        for k, c in zip(k_sizes, n_channels):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))
        self.pooling = nn.GlobalMaxPool1D()
        self.dropout_1 = nn.Dropout(0.3)
        self.dense_h = nn.Dense(6, activation='relu')
        self.dropout_2 = nn.Dropout(0.6)
        self.dense = nn.Dense(2)
        self.flat = nn.Flatten()

    def hybrid_forward(self, F, inputs, *args, **kwargs):
        emb = self.embedding(inputs)
        input_emb = emb.transpose((0, 2, 1))
        conv_process = [self.flat(self.pooling(conv(input_emb))) for conv in self.convs]
        dense_inputs = mx.sym.Concat(*conv_process, dim=1)
        hid = self.dense_h(self.dropout_1(dense_inputs))
        outputs = self.dense(self.dropout_2(hid))
        return outputs


def softmax(X):
    return X.exp() / X.exp().sum(axis=1, keepdims=True)


w2v_size, k_sizes, n_channels = 10, [3, 4, 5], [10, 10, 10]
in_vars = locals()
for i in labels_index:
    filename = './models_params_file/model_params_{0}'.format(labels_name[i-1])
    net = TextCNN(voc, w2v_size, k_sizes, n_channels)
    net.load_parameters(filename)
    net.hybridize()
    test_features_ndarrray = nd.array(test_features)
    in_vars['test_pre_{0}'.format(labels_name[i-1])] = net(test_features_ndarrray)
    in_vars['test_{0}_as'.format(labels_name[i-1])] = softmax(in_vars['test_pre_{0}'.format(labels_name[i-1])])

results_dict = {labels_name[i-1]: in_vars['test_{0}_as'.format(labels_name[i-1])][:, 1].asnumpy() \
                for i in labels_index}

pre_results = pd.DataFrame(data=results_dict)

results = pd.concat((test_data['id'], pre_results), axis=1)

results.to_csv('submission.csv', index=False)
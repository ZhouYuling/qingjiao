
#客户投诉分类
#1.文件读取和数据拆分。
#2.分词、去停用词。读取同目录下的停用词表，用结巴分词器对文本进行分词、去停用词。
#3.创建词表。遍历文本，以字典形式统计词频，将词在字典的位置映射为索引，并在字典中加入未知和
#填充符号。
#4.创建Dataset和Dataloader并实例化。
#5.搭建自己的TextCNN模型。根据前向传播和输出的模型结构搭建模型。
#6.模型训练验证测试。进行模型的训练评估，并查看在测试集上的效果。


#读取停用词
def stopwords():
    with open('HGD_Stopwords.txt', 'r', encoding='UTF-8') as f: # 读取停用词表中的停用网
        stopwords = ???(line.??? for line in f) # 使用集合存储停用词， 以便快速查找
    return stopwords

stop_words = stopwords()
stop_words
#修改为
stopwords = set(line.strip() for line in f) # 使用集合存储停用词， 以便快速查找


#知识点2：用结巴分词的精准模式进行分词，返回的是可迭代对象
# 分词、去停用词
def tokenizer(texts):
    word_cut[]
    # 遍历每个句子，对句子进行分词、去停用河
    for text in texts:
        words = [word for word in jieba.???(text,cut_all-???) if word not in stop_words] # 使用jieba进行精确模式分词(返回可迭代对象），并去除停用词
        word_cut.append(words)
    return word_cut
#tokenizer_text = tokenizer(texts)
#tokenizer_text [1]

#修改为：
words = [word for word in jieba.cut(text,cut_all=False) if word not in stop_words]


#知识点3：遍历文本，得到每个词出现的次数；枚举排序后的词语，将索引设为词汇表的索引，提取值的第一个元素（即词），构建一个词-id 映射表。将未知词和填充符号加入字典并赋予索引。
#创建 词-id 表， 使每个词都有唯一的数字映射， 并设置未知词和填充词
#1.创建字典，对每条文本中的词进行添加并计数，形成一个包含所有词的字典。
#2.按照词频对词语进行计数，选取词频大于N的词，保留出现次数最多的M个值。(N,M为自己设定的值)
#3.将单词在字典中的位置映射为 词-id 表的索引，并将未知词('<UNK>')和填充词(<PAD>)添加到词表。
#构建 词-id 表
MAX_VOCAB_SIZE=15000#感长词表
MIN_WORD_FREQ=5 #最小词频
UNK, PAD = '<UNK>','<PAD>' # 未知书，填充符号
def build_vocab(texts, tokenizer, max_size, min_freq):[
    # 定义一个空字厚，用干存储单词及其出现饮数
    vocab_dic = {}
    for line in tokenizer(texts):#对于每一行文本
        if not line: # 跳过空行
            continue
        for word in line:#对于每一个词
            # 用.get()方法获取 word 的值，如果该词语不存在I则返回默以值 e.
            vocab_dic[word] = vocab_dic.??? + 1 # 統计词出現次數，每次出現则加 1
    # 按照词出现次数从大到小排序，只保留出现次数最大的前max_size个单词
    vocab_list = sorted([_ for_in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    # 构建一个字牌，将词题射到它们在词汇表中的素引，word_count包音词和词找
    vocab_dic = {???: ??? for idx, word_count in enumerate(vocab_list)}
    #都加两个特殊词，用于在文本序列中填充或替代来知词
    vocab_dic.???({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

#通过满用 build_vocab 函数可以得到一个词汇表，将单词映射为它们在词汇表中的索引
vocab - build_vocab(texts, tokenizer, MAX_VOCAB_SIZE, MIN_WORD_FREQ)
vocab

#补充如下：
vocab_dic[word] = vocab_dic.get(word,0) + 1
vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list))
vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})



#知识点4：构造collate_batch函数，将每个batch_size中的序列填充为相同长度，填充值设置为 vocab['&lt;PAD&gt;'];
#知识点5：实例化DataLoader，设置batch_size，设置是否打乱（训练集打乱，测试集和验证集不打乱），设置自定义函数；
from torch.nn.utils.rnn import pad_sequence
"""
创建Dataset数据集和Dataloader数据加载器
1,创建数据集类,并在数据集内将文本转化为向里。
2.构造处理函数collate_fn，作用是对句子进行填充，使得每个batch内句子长度相同。
3.创建数据集实例。
4.设置batch size大小，创建数据加载毒实例。
"""
#创建Dataset 和Dataloader
class TextDataset(Dataset):
    def_init_(self, data):
        # 初始化函数，接受数据作为参数
        # 将输入文本的句子列表进行分泌，并将每个词转换为闪表中的id
        # 为方便直接在此处理
        self,sequences - [[vocab.get(word, vocab.get(UNK)) for word in sentence] for sentence in tokenizer(data['text'].values)]

        #保存标签
        self.labels = data['label'].values

        def _len_(self):
            # 返回数据樂的长度
            return len(self,sequences)

        def __getitem__(self, idx):
            # 根据索引获取数蕾
            #返回该素引对应的句子的词表id序列和标签
            return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

def collate_batch(batch):
    #定义用干将一个batch的样本组合成一个张量的函数
    label_list, text_list = [], []
    # 遍历每个batch的数基
    for _text, _label in batch:
        label_list.append(_label)
        text_list.append(_text.clone().detach())

    #对句子列表进行填无，使其长度一致
    text_list = ???(text_list, batch_first=True, ???=vocab['<PAD>'])
    label_list torch.tensor(label_list, dtype-torch.int64)

    # 返回填充后的句子列表和标签列表
    return text_list, label_list

#创建训练、测试和验证的数据藥尖例
train_data- TextDataset(train)# 训练
test_data = TextDataset(test) # 测试
valid_data = TextDataset(valid) # 短通

batch_size=32
#创建训练、测试和验证的DataLoader实例
train_loader - DataLoader(???, batch_size=batch_size, shuffle=???, ???) # 训练
test_loader = DataLoader(???, batch_size=batch_size, shuffle=???, ???) # 测试
valid_loader = DataLoader(???, batch_size=batch_size, shuffle=???, ???) # 验证

#修改为：
text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab['<PAD>'])
train_loader - DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch) # 训练
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch) # 测试
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch) # 验证


#知识点6：词向量层搭建（词表大小、词嵌入维度）；双向LSTM层搭建（词向量为度，隐藏单元维度，LSTM层数，使用双向，设置输入维度）
#知识点7: 1维卷积层（输入通道数，卷积核数量，卷积核大小) ；自适应池化层；线性层
"""
搭建自己的TextRCNN模型,进行参数配置。
网络层：
-Embedding层:将输入的文本索引转换为密集向里,输入为《词汇表大小,嵌入层维度) .
-双向LSTM网络:设置输入维度、隐藏状态的维度、层的数量、使用双向、输入的张里形状为(batch-size, seq_length, embedding-dim)
-一维卷积层:提取文本特征,输入通道数为embedding dim + hidden_size * 2,输出通道数为filters,卷积核大小已设置好。
-自适应最大池化层：输出尺寸为(batch_size, filters, 1)
-全连接层：输入特征数为 filters，输出特征数为 num_classes
前向传播：
（1）词向重层
（2）双向LSTM层
（3）拼接
（4）卷积
（5）池化
（6）全连接
"""
import torch
import torch.nn as nn

# 定义 TextRCNN 类，避承自 nn.Module
class TextRCNN(nn.Module):
    def _init_(self, vocab_size, embedding_dim, hidden_size, filters, num_layers, kernel_size, num_classes):
        super(TextRCNN, self).__init()

        # 词向量房
        self.embedding = nn.???

        #双向 LSTM 层，输入为 embedding_dim，输出为 hidden_size*2
        # batch_first 表示输入的第一籮是 batch_size, 即 (batch_size, seq_len, input_size)
        self.bidirectional_rnn = nn.???

        # 卷积度,输入通道数为 embedding_dim + hidden_size * 2, 输出通道数为 filters道
        self.conv = nn.???

        #自适应池化层，输出尺寸为(batch_size, filters, 1)
        self.pool = nn.???(1)
        # 全连接展，输入特征数为 filters， 输出特征数为 num_classes
        self.fc = nn.???(filters, num_classes)

    def forward(self, inputs):
        # 前向传播过程
        x = self.embedding(inputs) # 词向重层
        rnn_output, _ self.bidirectional_rnn(x) # LSTM 展
        combined = torch.cat((x, rnn_output), dim=2) # 拼接词向量和 LSTM 输出，形成 [botch_size, seq_len, embedding_dim + 2 * hidden_size] 的张量l
        conv_output = self.conv(combined.permute(0, 2, 1)) # 卷积层
        pooled_output self.pool(conv_output).squeeze(2) # 油化展
        output = self.fc(pooled_output) # 全连锁厚
        return output

# 機型參数
vocab_size = len(vocab) # 阀汇表大小
embedding_dim = 100 # 词向量维度
hidden_size -64 # RNN 单元数
filters = 64 # 卷积核数目
num_layers = 2 # LSTM 层数
kernel_size = 3 # 卷积技大小
num_classes = len(set(labels)) # 分类类别数

#修改为：
self.embedding = nn.Embedding(vocab_size, embedding_dim)
self.bidirectional_rnn nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)
self.conv = nn.Conv1d(embedding_dim + 2 * hidden_size, filters, kernel_size)
self.pool = nn.AdaptiveMaxPool1d(1)
self.fc = nn.Linear(filters, num_classes)



#知识点8:选用Adam优化器，学习率设置为0.01和L2正则化为0.01;损失函数为多分类的交叉熵损失。
#知识点9：设置训练模式，梯度归零，单向传播，梯度更新。
#知识点10:设置评估模式,关闭梯度计算。
import torch.optim as optim
# 训练函数，用干训练调参
"""
模型训练

训练：
用训练集进行训练,计算每个epoch的准确率和损失。
验证：
用验证集做验证,计算每个训练epoch下的准确率和损失。
测试：
用测试集做测试，输出最终结果。
"""
#用CPU还是GPU
device = torch.device('cuda' if torch.cuda.is_available()Telse 'cpu')
optimizer = optim.???(model.parameters(), Ir=0.01, weight_decay=0.01)
criterion = nn.???

def train(model, train_loader, valid_loader, test_loader, num_epoch):
    #交叉嫡损失
    criterion = nn.CrossEntropyLoss()
    #设置训练轮片
    model = model.to(device)
    for epoch in range(num_epoch):
        # 训练阶段
        ???
        # 初始化训练编损失函数和正确预测个数
        train_loss = 0.0
        train_corrects 0.0
        #遍历训练果
        for texts, labels in train_loader:

            # 将数据移动到GPU
            labels = labels.to(device)
            texts = texts,to(device)

            optimizer.??? # 梯度归常
            outputs = model(texts) #前向传播得到模型输出
            loss = criterion(outputs, labels) # 计t算损失

            loss.??? # 反向传播计算梯度
            optimizer.??? # 梯度更新

            train_loss += loss.item() # 损失函数票加

            _,predicted = torch.max(outputs.data, 1) # 找到预测結樂中的展大值
            train_corrects = (predicted == labels).sum().item() # i算正确预测个数

        train_size - len(train_loader.dataset) # troin_loader大小
        train_loss /= train_size # 平均损失
        train_acc = 100.0 * train_corrects / train_size # 在糖率

        # 在验证藥上进行验证
        val_acc, val_loss = evaluate(model, valid_loader, criterion)

        # 打印当前epoch的训练和验证結果
        print('Epoch: {} \n Train Loss: {} Train Accuracy: {}% \t'.format(epoch+1, train_loss, train_acc),
            'Valid Loss: {} Valid Accuracy: {}%',format(val_loss, val_acc))

    # 在测试架上进行测试
    test(model, test_loader)

#验证函数
def evaluate(model, valid_loader, criterion):
    val_loss =0.0 # 累汁验证樂损失
    val_corrects =0.0 # 累汁验证集准确率
    # 评估模式
    ???

    with torch.???: # 关闭梯度计算
        for texts, labels in valid_loader:

            texts, labels = texts.to(device), labels.to(device) # 符数据p动到GPU
            outputs = model(texts) #前向传播得到模型输出
            loss = criterion(outputs, labels) # 计算损失

            val_loss += loss.item() # 损失函数参加

            _, predicted = torch.max(outputs, 1) # 找到预测结果中的网大值
            val_corrects += (predicted == labels).sum().item() # 计算正确颜测的个数

        val_size = len(valid_loader.dataset) # val_looder大小
        val_loss /= val_size # 平均损失
        val_acc = 100.0 * val_corrects/val_size # 准鹅率
    return val_acc, val_loss # 返回平均准确奉和平均损关

# 测试函数
def test(model, test_loader):
    model.eval() #设置模型为评估模式
    test_acc, test_loss = evaluate(model, test_loader,criterion)# 调用evaluate函数计算测试柴上的在确率和损头
    print('Test Loss: {} Test Accuracy: {}%'.format(test_loss, test_acc))



#修改为
optimizer = optim.Adam(model.parameters(), Ir=0.01, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
# 训练阶段
model.train()
optimizer.zero_grad() # 梯度归常
loss.backward() # 反向传播计算梯度
optimizer.step() # 梯度更新
# 评估模式
model.eval()
with torch.no_grad(): # 关闭梯度计算



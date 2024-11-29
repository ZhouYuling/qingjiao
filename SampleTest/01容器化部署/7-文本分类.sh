#智侦反诈分类引擎

#1.读取数据文件，查看前10行
#2.查看数据详情，对类别标签进行去重并统计分析情况
#3.检测缺失值和重复值并处理



#考点1:使用pandas读取"/home/qingjiao/TelecomFraudProject/data/"目录下的电信诈骗数据文件(注意停用词和模型权重也在相应目录下) ,查看前10 行数据。
#数据资源鑰径
data_path = ??? # 训练集数還
stopwords_path = ??? # 停用词
model_weights = ??? # 仅置位置

#读取数据
data = pd.read_csv(???, encoding = 'UTF-8') #读取csv数据文件，编网格式为utf-8
data.head(???) #直看前10行

#查看数据详情与类型

#考点2：使用 pandas 一个函数查看数据详情。
#查看数据详细信息
data.???

#考点3：对“案件类别”进行去重，统计总类别数。
#查看有多少不同的类
label = list(data['案件类别'].???)
# 计算出类别数
num_category = ???(label)
print(label,"总类别数：",num_category)# 打印标签类别

#统计类别标签的分布情况
#考点4：统计各"案件类别"的分布情况。
data.groupby(['案件类别']).??? # 查看案件类别的分布

#检查缺失值并处理
#考点5：检查"案情描述”中是否存在缺失值，如果存在缺失值则删除，并输出缺失值数量。
#检查“案情描述”列的缺失值数量
missing_values = data['案情描述'].???.???
# 馨出缺失值的行数（非常字
missing_rows missing_values[missing_values > 0]
print("缺失值数里：",missing_rows.???)

#检查重复值并处理
#考点6：检查"案情描述”中是否存在重复值，如果存在重复值则删除，并输出重复值数量。
#检查“案情描述” 列是否存在董复值
duplicate_rows = data['案情描述'].???.???
print("重复值数里：", duplicate_rows)
# 刪除軍复值
data.???(subset=['案情描述'], keep='first', inplace-True)

#文本数据处理
#1.类别标签数值索引化
#2.jieba 分词处理
#3.停用词处理
#4.高频词可视化

#类别标签数值化
#考点7：将"案件类别"标签列映射为数值索引,并输出标签数量。
# 增娜标签列
def label_dataset(row):
    num_label = label.???(row) #提取Label的索引
    return num_label

data['label'] = data[???].apply(label_dataset) # 根据"案件类别" 索引生成类别标签
labels = data['label'].tolist() # 转换为 List 列表

#直看标签数量
print("标签数重：",len(labels))

#jieba 分词
#考点8:使用jieba 库完成对文本的分词,封装为函数。
#定义中文分词函数

def chinese_word_cut(row):
    # 先进行结巴分词
    return list(jieba.???(row)) # 返回分词结果的列表

#停用词处理
#1.白定义加载停用词函数
#2.自定义去除特殊字符和数的函数
#考点9:自定义加载停用词列表、去除特殊字符和数值和去除停用词的函数。
#实现加裁停用词列表函数
def load_stopwords_list(filepath):
    stopwords = []
    with open(filepath, 'r', encoding='utf-8') as reader_stream:
        for line in reader_stream:
            stopwords.append(line.strip()) # 去掉行末的部行符
    return stopwords


# 去除特殊符号和数值的函数
def remove_punctuation(line):
    text_line = ???(line)
    if text_line strip() == '':
        return
    text_line = re.???(???, '', text_line) # 去臉母务句子中的数值
    rule = re.???(u"[^a-ZA-Z0-9\u4E00-\u9FA5]") # 根据定义的规则去除无用符号
    text_line = rule.???('', text_line)
    return text_line

#去除停用词的函岩
def remove_stopwords(tokenized_texts, stopwords):
    return [word for word in tokenized_texts if word not in stopwords and word]

#考点10:调用分词、停用词和特殊字符处理函数，获取处理后列表。
#应用函数实现中文分泌并将其转换为列表
tokenized_texts = data['案情描述'].apply(chinese_word_cut).??? # 对每行进行分词
#加裁停用词列表
stopwords = load_stopwords_list(???)

# 去除特殊符号并处理停用沟
cleaned_texts = []
for tokens in ???:
    tokens = [remove_punctuation(token) for token in tokens] # 去除标点
    cleaned_tokens = remove_stopwords(tokens, stopwords) # 去除停用网
    cleaned_texts.append(cleaned_tokens)

print("经分词和停用词后的词表：", cleaned_texts[:5])# 输出经过分词和停用词后的 5 个词表
#绘制词汇分布的词云图
#考点11:使用WordCloud库绘制最高出现次数的前50个词的词云图。d
#过滤掉停用词和空中药南
#続计词频
all_words = [word for tokens in cleaned_texts for word in tokens] #扁平化
word_counts = ???(all_words) # 统计每个词的频率
most_common_words = dict(word_counts.most_common(???)) # 获取前10个高顿词
#绘制润云图
wordcloud = WordCloud(font_path='/usr/share/fonts/chinese/SimHei.ttf', # 篇要提供中文中体的路径
                        width=800,
                        height=400,
                        background_color='white').generate_from_frequencies(???)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')# 不显示坐标轴
plt.title('词汇分布的词云图')
plt.show()

#特征工程
#1.构建词典
#2.序列映射
#3.划分数据集
#4.转换数据类型

#构建词典，统一序列长度
#考点12：自定义函数 build_vocab_and_sequences，实现生成文档词典和统一词向量序列的长度。
#自定义函數来拘建词表和填充序列
def build_vocab_and_sequences(tokenized_texts):
# 扁平化列表以收集所有词
    flat_list = [word for sublist in tokenized_texts for word in sublist]
    #生成词汇表，包含所有唯一词汇
    vocab = (word: idx + 1 for idx, word in ???(set(flat_list)))
    vocab['<PAD>'] =Θ # 添加填充标记

    # 符网语转换为整数索引
    sequences = [[vocab.get(word, vocab['<PAD>']) for word in text] for text in tokenized_texts]

    #计算序列的均值长度
    lengths = [len(seq) for seq in sequences)
    mean_length = int(sum(lengths) ??? len(lengths)) if lengths else o # 防止除以零

    #設定最大序列长度为均值长度
    max_length = mean_length

    #填充或裁新序列以达到最大长度
    padded_sequences = []
    for seq in sequences:
        if len(seq) ??? max_length:
            # 森断，保留中间部分
            start = (len(seq) - max_length) // 2
            end = start + max_length
            padded_sequences.append(seq start:end])

        else:
            #填充
            padding = [vocab['<PAD>']] * (max_length - len(seq))
            padded_sequences,append(seq + padding)

    return vocab, padded_sequences, max_length

#考点13:调用生成文档词典和统一序列长度函数,输出词表、序列长度和第一个序列。
#满用函数构建调表和序列
vocab, padded_text_sequences, max_length = build_vocab_and_sequences(cleaned_texts)
# 查看词表、最长序列、填充后的序列
print("词表:", list(vocab.???)[0:10])
print("序列长度：", max_length)
print("第一个序列：", padded_text_sequences[0:1])

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
???,
???,
test_size=???,
random_state=42
)
#转换数据类型
# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
#模型配置
#1.设定超参数
#2.定义数据集类
#3.构建训练集和测试集加载器

#定义超参数
#考点15:定义超参数，分别设置词长度、嵌入层维数、隐藏层维数、总类别数、批大小和训练轮数。
# 定义模型参数
vocab_size = len(vocab) # 网长度
embedding_dim = 256 # 飲入厚燃影
hidden_dim = 512 # 隐藏展维数
num_classes = ??? #总类别数
batch_size = 128 # 输入船大小
epochs = 1 # 训练彩数
#构建自定义数据集类
class TextDataset(Dataset):
    def __init_(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len_(self):
        return len(self.texts)

    def __getitem_(self, idx):
        return self.texts[idx], self.labels[idx]

#创建训练集和测试集数据加载器
#考点16：构建训练集和测试集的数据加载器。
# 封裝训练藥和测试藥
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
# 构建训练藥和测试樂影量加貌器
train_loader = DataLoader(???, batch_size-batch_size, shuffle-True)
test_loader = DataLoader(???, batch_size-batch_size)

#模型训练
#1构建RNN + Attention循环神经网络
#2.设置计算平台和优化器
#3.训练模型

#构建RNN+Attention循环神经网络模型
#考点17:定义双向 RNN 神经网络模型，并引入 Attention 机制，捕捉重要特征。
class RNNAttentionModel(nn.Module):
    def __init_(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNNAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=???, bidirectional=???)
        self.attention = nn.Linear(hidden_dim 小于小于 1, 1)
        self.fc = nn.Linear(hidden_dim 小于小于 2, ???)
        self.dropout = nn.Dropout(???)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out,self.rnn(embedded)

        # 注意力机制
        attn_weights = F.softmax(self,attention(rnn_out),squeeze(-1), dim-1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)

        # 连接RNN和DAttention
        x = torch.cat((rnn_out[:, -1, :], attn_applied), dim-1)
        self.fc(self.dropout(x))
        return x

#实例化模型
model = RNNAttentionModel(vocab_size, embedding_dim, hidden_dim, num_classes)
#加裁预训练衣置
model.load_state_dict(torch.load(???, map_location=???))

#设置计算平台和优化器
#考点18：判断是否有可用GPU资源，定义交叉熵损失函数和构建 AdamW 优化器，设置学习率和惩罚。
#转節模型计算平台
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
#定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#构建优化器
optimizer = torch.optim.Adam(model.parameters(), Ir=???, weight_decay=???)

#自定义训练模型函数
#考点19：自定义训练模型函数，启用训练模式，输入数据，计算损失和打印训练集预测准确率，注意清空梯度。
#训练横型
def train model(model, train_loader, criterion, optimizer, device):
    model.train() #启用训练模式
    running_loss = 0.0
    correct_predictions = 0 # 正确预测的数量
    total_samples=0 #样本数

    for i, (texts, labels) in enumerate(train_loader):
        texts, labels - texts.to(device), labels.to(device) # 如果有GPU则在GPU上运行
        optimizer.??? # 梯度归零
        outputs = model(texts) #文本输入模型
        loss = criterion(???, ???) # 计算损失
        loss.??? # 反向传播
        optimizer.??? #更新模型参数
        running_loss += loss.item() # 絮加损失值


#考点20：自定义评估函数，在测试集上进行预测推理，保存预测值和真是标签值列表，输出准确率，返回输出和标签。
# 评估摸型
def evaluate_model(model, test_loader, device):
    model.eval() #启用评估模式
    all_preds = [] # 定义预斜值列表
    all_labels = [] # 定义标签值列表
    all_outputs [] # 保存所有輸出以便计算ROC曲线

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device) # 如果有GPU则在GPU上运行
            outputs = model(texts) # 测试樂输入模型
            all_outputs.append(outputs.cpu().numpy()) # 保存输出
            _, preds = torch.max(outputs, dim=???) # 获取预测值
            all_preds.extend(???)
            all_labels.extend(???)


    accuracy - accuracy_score(all_labels, all_preds)
    print(f'Test-Accuracy: {accuracy *100:.2f)%') # 仅打印在确率

    #保存输出和标签以便后续计算情确率，召回率和F1分数
    return np.vstack(all_outputs), np.array(all_labels)

#计算评估指标
#考点21:自定义计算评估指标函数，通过评估函数返回的输出和标签计算精确率、召回率和F1分数，并绘制逐类的 ROC 曲线。
#计算并打印评估指标
def compute_metrics(all_outputs, all_labels):
    # 确保 all_outputs 是一个二雄制格式
    if all_outputs.ndim == 1:
        all_outputs = np.expand_dims(all_outputs, axis=1) # 将一维数组扩展为二繼

    all_labels_binarized -label_binarize(all_labels, classes-np.arange(all_outputs.shape[1])) # 二值化标签
    n_classes = all_labels_binarized.shape[1]

    # 计算每个类别的精绪率、日回率和F1分数
    precision = precision_score(all_labels_binarized, (all_outputs &gt; 0.5).astype(int), average='weighted', zero_division-e)
    recall = recall_score(all_labels_binarized, (all_outputs &gt; 0.5)astype(int), average-'weighted', zero_division-0)
    f1 - f1_score(all_labels_binarized, (all_outputs &gt; 0.5).astype(int), average-'weighted', zero_division-0)

    #打印精确率、召回率和F1分数
    print(f'Precision: (precision * 100:.2f)%')
    print(f'Recall: (recall 100:.2f)%')
    print(f'F1 Score: (f1 *100:.2f)%')

    #绘制每个类别的 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _= roc_curve(all_labels_binarized[:, 1], all_outputs[:, 1]).
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label-f'Class (4) (AUC - (roc_auc:.2f))')

    plt.plot([0, 1], [0, 1],'k--') #参考线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt. show()

#执行训练和评估
#主训练痛环
for epoch in range(epochs):
    training_loss, training_accuracy = train_model(model, train_loader, criterion, optimizer, device)
    print(f'Epoch: (epoch + 1)， Loss: (training_loss:.8f}，Train-Accuracy: (training_accuracy:.4f)') # 打印轮数、损失值和准确率

#评估模型并计算指标
all_outputs, all_labels = evaluate_model(model, test_loader, device)
compute_metrics(???,???) # 计算并打印精晴率，召回率和F1分数

#修改为：
data_path = "./data/fraud_data.csv"
stopwords_path = "./stopwords/stopwords.txt" # 停用词
model_weights = "./weights/model_weights.pth"
data = pd.read_csv(data_path, encoding = 'UTF-8') #读取csv数据文件，编网格式为utf-8
data.head(10) #直看前10行
data.info()
label = list(data['案件类别'].unique())
num_category = len(label)
data.groupby(['案件类别']).size # 查看案件类别的分布
missing_values = data['案情描述'].isnull().sum()
print("缺失值数量：",missing_rows.shape[0])
duplicate_rows = data['案情描述'].duplicate().sum()
data.drop_duplicates(subset=['案情描述'], keep='first', inplace=True)
num_label = label.index(row) #提取Label的索引
data['label'] = data['案件类别'].apply(label_dataset) # 根据"案件类别" 索引生成类别标签
return list(jieba.lcut(row)) # 返回分词结果的列表
text_line = str(line)
text_line = re.sub(r'\d+', '', text_line) # 去臉母务句子中的数值
rule = re.compile(u"[^a-ZA-Z0-9\u4E00-\u9FA5]") # 根据定义的规则去除无用符号
text_line = rule.sub('', text_line)
tokenized_texts = data['案情描述'].apply(chinese_word_cut).tolist() # 对每行进行分词
stopwords = load_stopwords_list(stopwords_path)
for tokens in tokenized_texts:
word_counts = Counter(all_words) # 统计每个词的频率
most_common_words = dict(word_counts.most_common(10)) # 获取前10个高顿词
background_color='white').generate_from_frequencies(most_common_words)
vocab = (word: idx + 1 for idx, word in enumerate(set(flat_list)))
mean_length = int(sum(lengths) / len(lengths)) if lengths else o # 防止除以零
if len(seq) > max_length:
print("词表:", list(vocab.keys)[0:10])
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
padded_text_sequences,
labels,
test_size=0.2,
random_state=42
)
num_classes = num_category #总类别数
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size-batch_size)
self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
self.fc = nn.Linear(hidden_dim '<<' 2, num_classes)
self.dropout = nn.Dropout(0.2)
model.load_state_dict(torch.load(model_weights, map_location='cpu'))
optimizer - torch.optim.Adam(model.parameters(), Ir=0.0001, weight_decay=0.01)
optimizer.zero_grad() # 梯度归零
loss = criterion(outputs, labels) # 计算损失
loss.backward() # 反向传播
optimizer.step() #更新模型参数
_, preds = torch.max(outputs, dim=1) # 获取预测值
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
compute_metrics(all_outputs,all_labels) # 计算并打印精晴率，召回率和F1分数


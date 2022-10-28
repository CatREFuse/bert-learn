```python
import pandas as pd
from tqdm import tqdm
import os
```


```python
def save(path: str, content: str) -> None:
    # if dir doesn't exist, create dir
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file = open(path, "w")
    file.write(content)
```


```python
with open("./baike_qa_valid.json", "r") as file:
    res = ["[\n"]
    for line in tqdm(file.readlines()):
        content = line[:-1] + ",\n"
        res.append(content)
    res[-1] = res[-1][:-2] + "\n"
    res.append("]")
    save("./baike.json", "".join(res))
```

    100%|██████████| 44972/44972 [00:00<00:00, 2058429.43it/s]


## token 化字符串

使用 `BertTokenizer` 的预训练模型 `bert-base-chinese` 将字符串转换为 token 序列。

首先安装这个与训练模型

```shell
git clone https://huggingface.co/bert-base-chinese
```


```python
# tokenizer
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

example_text = '屁股上长了个痣，是什么病？'

bert_input: torch.Tensor = tokenizer(example_text, padding="max_length", max_length=24, truncation=True, return_tensors="pt")

print(bert_input["input_ids"])
print(bert_input["token_type_ids"])
print(bert_input["attention_mask"])
```

    tensor([[ 101, 2230, 5500,  677, 7270,  749,  702, 4582, 8024, 3221,  784,  720,
             4567, 8043,  102,    0,    0,    0,    0,    0,    0,    0,    0,    0]])
    tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


- `input_ids` 代表了句子中每个词的 token。

- `token_type_ids` ，它是一个二进制掩码，用于标识令牌所属的序列。如果我们只有一个序列，那么所有的 token 类型 id 都将为 0。对于文本分类任务，token_type_ids 是我们 BERT 模型的可选输入。

- `attention_mask` ，它是一个二进制掩码，用于标识标记是真实单词还是只是填充。如果 token 包含 [CLS]、[SEP] 或任何真实单词，则掩码将为 1。同时，如果令牌只是填充或 [PAD]，则掩码将为 0。


```python
tokenizer.decode(bert_input["input_ids"][0])  # 解码成原文
```




    '[CLS] 屁 股 上 长 了 个 痣 ， 是 什 么 病 ？ [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'




```python
# 读数据
dataset = pd.read_json("./baike.json")
```


```python
dataset["category"].value_counts()
```




    游戏-网络游戏           3086
    娱乐-博彩             1576
    烦恼-恋爱             1279
    电脑/网络-互联网-上网帮助    1277
    商业/理财-股票          1268
                      ... 
    电脑/网络-百度-百度知道        1
    医疗健康-五官科-耳鼻喉科        1
    医疗健康-妇产科-产科          1
    游戏-腾讯游戏-英雄联盟         1
    娱乐休闲-收藏              1
    Name: category, Length: 321, dtype: int64




```python
# category column reserve two char
dataset["category"] = dataset["category"].apply(lambda x: x[:2])
# remove item with category is ""
dataset = dataset[dataset["category"] != ""]
dataset["category"].value_counts()
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>qid</th>
      <th>category</th>
      <th>title</th>
      <th>desc</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>qid_1815059893214501395</td>
      <td>烦恼</td>
      <td>请问深入骨髓地喜欢一个人怎么办我不能确定对方是不是喜欢我，我却想</td>
      <td>我不能确定对方是不是喜欢我，我却想分分秒秒跟他在一起，有谁能告诉我如何能想他少一点</td>
      <td>一定要告诉他你很喜欢他 很爱他!!  虽然不知道你和他现在的关系是什么！但如果真的觉得很喜欢...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>qid_2063849676113062517</td>
      <td>游戏</td>
      <td>我登陆诛仙2时总说我账号密码错误，但是我打的是正确的，就算不对我?</td>
      <td></td>
      <td>被盗号了~我的号在22号那天被盗了，跟你一样情况，link密码与账号错误，我密保都有了呐，邮...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>qid_6625582808814915192</td>
      <td>游戏</td>
      <td>斩魔仙者称号怎么得来的</td>
      <td>斩魔仙者称号怎么得来的</td>
      <td>楼主您好，以下为转载：\r\r圣诞前热身 来《生肖传说》做斩魔仙者\r\r　　一年一度的圣诞...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>qid_9204493405205415849</td>
      <td>商业</td>
      <td>有哪位好心人上传一份女衬衫的加拿大海关发票给我看一下塞多谢了</td>
      <td>多谢了</td>
      <td>我给你信息了</td>
    </tr>
    <tr>
      <th>4</th>
      <td>qid_5049427108036202403</td>
      <td>娱乐</td>
      <td>想去澳州看亲戚二个星期，怎么去，求教</td>
      <td>想去澳州看亲戚二个星期，怎么去，求教</td>
      <td>你看亲戚，申请的是旅游签证676！澳洲旅游签证很容易的。 \r\n你让亲戚将他的护照签证页和...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>44967</th>
      <td>qid_2831512218652800922</td>
      <td>生活</td>
      <td>房贷事宜我想咨询一下本人家在温州苍南钱库镇上有一套房子，市值二十</td>
      <td>我想一下本人家在温州苍南钱库镇上有一套房子，市值二十万左右，可否到杭州银行贷款，需提供哪些证件，</td>
      <td>你好!不可以!不是同一市的房产不可以贷款!你温州的房产,要到温州的银行申请抵押贷款!</td>
    </tr>
    <tr>
      <th>44968</th>
      <td>qid_674883050627329082</td>
      <td>健康</td>
      <td>为何会做梦?我已经有7~8年的做梦经历了,天天做梦,而我才25岁</td>
      <td>我已经有7~8年的做梦经历了,做梦,而我才25岁.而且近几年还变本加利,还讲梦话,为什么会这...</td>
      <td>您好！做梦是人在睡眠过程中产生的一种正常的生理和心理现象。当人在深度睡眠的时候，大脑神经细胞...</td>
    </tr>
    <tr>
      <th>44969</th>
      <td>qid_9172175649428541931</td>
      <td>娱乐</td>
      <td>交通事故赔偿车祸使我在急诊室里住了五天，可以要求赔偿伙食费吗？</td>
      <td>车祸使我在室里住了五天，可以要求赔偿伙食费吗？</td>
      <td>您好。\r\n\r\n如果有关的证据是可以的。</td>
    </tr>
    <tr>
      <th>44970</th>
      <td>qid_715056891459073707</td>
      <td>商业</td>
      <td>市盈率的多少是什么意思?</td>
      <td></td>
      <td>市盈率是股价除以股票的年净利润,其实多少倍的市盈率就是每只股票多少年能收回成本!市盈率越高,...</td>
    </tr>
    <tr>
      <th>44971</th>
      <td>qid_1226040859075871464</td>
      <td>健康</td>
      <td>孕妇如何办理产检我老婆现在怀孕四个月了，我是上海户口，我老婆是外</td>
      <td>我老婆现在怀孕四个月了，我是上海户口，我老婆是外地户口，请问去做产检，需要些什么程序，需要哪...</td>
      <td>我同学也是外地，嫁的上海人，她去年做产检的时候，当时是听她说回户籍所在地办啥手续，再在上海建...</td>
    </tr>
  </tbody>
</table>
<p>44924 rows × 5 columns</p>
</div>




```python
label_list = dataset["category"].unique().tolist()
labels = {label: i for i, label in enumerate(label_list)}
labels
```




    {'烦恼': 0,
     '游戏': 1,
     '商业': 2,
     '娱乐': 3,
     '生活': 4,
     '教育': 5,
     '育儿': 6,
     '健康': 7,
     '文化': 8,
     '电脑': 9,
     '社会': 10,
     '电子': 11,
     '体育': 12,
     '汽车': 13,
     '资源': 14,
     '医疗': 15}




```python
import torch
import numpy as np
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.labels = [labels[label] for label in dataset["category"]]
        self.questions = [tokenizer(text,
                                    padding="max_length",
                                    max_length=512,
                                    truncation=True,
                                    return_tensors="pt")
                          for text in dataset["title"]]
        # 不要忘记 return_tensors="pt"
    
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, index):
        return np.array(self.labels[index])

    def get_batch_questions(self, index):
        return self.questions[index]
    
    def __getitem__(self, index):
        return self.get_batch_questions(index), self.get_batch_labels(index)
```


```python
# 训练集:验证集:测试集 = 8:1:1
np.random.seed(42)
dataset_train, dataset_validate, dataset_test = np.split(dataset.sample(frac=1, random_state=42), [int(.8*len(dataset)), int(.9*len(dataset))])
print(len(dataset_train), len(dataset_validate), len(dataset_test))
```

    35939 4492 4493



```python
from torch import nn
from transformers import BertModel

class BertClassifer(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifer, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, len(labels))
        self.relu = nn.ReLU()
    
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_id, attention_mask=mask, return_dict=False)  # 不加 return_dict=False 会报错
        output = self.dropout(pooled_output)
        output = self.linear(output)
        output = self.relu(output)
        return output
```


```python
from pickletools import optimize
from torch.optim import Adam
from tqdm import tqdm

# get device ojbect according to the system
def get_device() -> str:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()

def train(model: torch.nn.Module, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)

            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(1) == train_label).sum().item()
            total_acc_train += acc

            # optimizer.zero_grad()
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        for val_input, val_label in val_dataloader:

            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)

            batch_loss = criterion(output, val_label.long())
            total_loss_val += batch_loss.item()
            
            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_acc_val += acc
            
            print(
                f'Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

EPOCHS = 5
model = BertClassifer()
LR = 1e-6

train(model, dataset_train, dataset_validate, LR, EPOCHS)

# 保存模型
torch.save(model.state_dict(), "model.pt")

```


```python

```

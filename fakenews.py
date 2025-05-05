

"""直接做BERT 訓練"""

from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# 讀取 Tokenizer
model_name = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# 讀取數據
df = pd.read_csv("test.csv")
df = df[df["label"].isin([0, 1])]
df["claim_words"] = df["claim_words"].apply(eval)
texts = [" ".join(words) for words in df["claim_words"]]
labels = df["label"].tolist()

# 自訂 Dataset
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], is_split_into_words=True,padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 準備 DataLoader
dataset = FakeNewsDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 設定模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 設定 Loss 和 Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 訓練
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 儲存模型
model.save_pretrained("bert_fake_news_model2")
tokenizer.save_pretrained("bert_fake_news_model2")




from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score

# 設定設備
device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取 Tokenizer 和 模型
model_name_test = "hfl/chinese-bert-wwm"
tokenizer_test = BertTokenizerFast.from_pretrained("bert_fake_news_model")
model_test = BertForSequenceClassification.from_pretrained("bert_fake_news_model").to(device_test)
model_test.eval()  # 設定為評估模式

# 讀取測試數據
df_test = pd.read_csv("test3.csv")
df_test = df_test[df_test["label"].isin([0, 1])]
df_test["claim_words"] = df_test["claim_words"].apply(eval)
texts_test = [" ".join(words) for words in df_test["claim_words"]]
labels_test = df_test["label"].tolist()

# 自訂 Dataset（新的變數）
class FakeNewsTestDataset(Dataset):
    def __init__(self, texts_test, labels_test, tokenizer_test, max_length_test=512):
        self.texts_test = texts_test
        self.labels_test = labels_test
        self.tokenizer_test = tokenizer_test
        self.max_length_test = max_length_test

    def __len__(self):
        return len(self.texts_test)

    def __getitem__(self, idx):
        encoding_test = self.tokenizer_test(
            self.texts_test[idx], padding="max_length", truncation=True, max_length=self.max_length_test, return_tensors="pt"
        )
        item_test = {key: val.squeeze(0) for key, val in encoding_test.items()}
        item_test["labels"] = torch.tensor(self.labels_test[idx], dtype=torch.long)
        return item_test

# 建立測試 DataLoader
test_dataset_new = FakeNewsTestDataset(texts_test, labels_test, tokenizer_test)
test_dataloader_new = DataLoader(test_dataset_new, batch_size=16, shuffle=False)

# 測試模型準確率
all_preds_new, all_labels_new = [], []

with torch.no_grad():
    for batch_test in test_dataloader_new:
        labels_test_batch = batch_test["labels"].numpy()
        batch_test = {k: v.to(device_test) for k, v in batch_test.items()}
        outputs_test = model_test(**batch_test)
        preds_test = torch.argmax(outputs_test.logits, dim=1).cpu().numpy()
        
        all_preds_new.extend(preds_test)
        all_labels_new.extend(labels_test_batch)

# 計算準確率（不覆蓋原變數）
accuracy_new = accuracy_score(all_labels_new, all_preds_new)
print(f"Test Accuracy (New Variables): {accuracy_new:.4f}")


#測試

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **載入 BERT Tokenizer**
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

# **定義與訓練時相同的模型架構**
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 768 → 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量
        logits = self.classifier(cls_embedding)  # 送入分類層
        return logits, cls_embedding

# **載入已儲存的模型**
model = BertClassifier(model_name).to(device)

# **載入模型權重**
model.load_state_dict(torch.load("bert_fake_news_model.pth", map_location=device))
model.eval()  # 設定為評估模式

# **定義預測函數**
def predict(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids, attention_mask = tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

    with torch.no_grad():
        logits, cls_embedding = model(input_ids, attention_mask)

    pred_label = torch.argmax(logits, dim=1).item()
    return pred_label, cls_embedding.cpu().numpy()

# **測試新聞**
test_text = "要到韓國遊玩的民眾注意！行動電源除了不可托運之外，還需以「絕緣膠帶包覆」或者裝入「透明密封的夾鏈袋」。不過，現在又多了一項新規定，行動電源若無標註「Wh」（瓦特小時，電池容量）或者數值高於「100Wh」都可能遭到海關沒收，建議大家出國前仔細檢查，避免憾事發生。"
pred_label, embedding = predict(test_text)

# **輸出結果**
print("預測結果:", "假新聞" if pred_label == 1 else "真新聞")


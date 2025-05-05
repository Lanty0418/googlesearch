# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 23:13:31 2025

@author: lanty
"""

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
test_text = "截止至2020年9月，中国研发的新冠疫苗已经有5个进入III期临床试验。​"
pred_label, embedding = predict(test_text)

# **輸出結果**
print("預測結果:", "假新聞" if pred_label == 1 else "真新聞")




"""XG模型測試"""
import torch
import xgboost as xgb
import numpy as np
from transformers import BertTokenizer, BertModel

# 加載預訓練的 BERT 模型和 tokenizer（這不需要額外儲存）
def load_bert_model():
    model_name = "bert-base-chinese"  # 使用預訓練的 BERT 模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # 設置模型為評估模式
    return model, tokenizer

# 加載已保存的 XGBoost 模型
def load_xgboost_model(model_path="xgb_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

# 取得 BERT 向量
def get_bert_embedding(text, model, tokenizer, device):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    
    with torch.no_grad():
        outputs = model(**tokens)

    # 使用 [CLS] token 作為句子向量
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# 即時測試函數
def test_text_function(text, model, tokenizer, xgb_model, device):
    # 1. 取得 BERT 向量
    embedding = get_bert_embedding(text, model, tokenizer, device)

    # 2. 用 XGBoost 模型進行預測
    prediction = xgb_model.predict(np.array([embedding]))  # 預測結果為 0 或 1
    return prediction[0]

# 設置設備（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載模型
bert_model, tokenizer = load_bert_model()  # 這會加載預訓練的 BERT 模型
xgb_model = load_xgboost_model(model_path="xgb_model.json")  # 加載保存的 XGBoost 模型

# 測試一個文本
test_text = "要到韓國遊玩的民眾注意！行動電源除了不可托運之外，還需以「絕緣膠帶包覆」或者裝入「透明密封的夾鏈袋」。不過，現在又多了一項新規定，行動電源若無標註「Wh」（瓦特小時，電池容量）或者數值高於「100Wh」都可能遭到海關沒收，建議大家出國前仔細檢查，避免憾事發生。"  # 這裡可以替換成任何你想測試的文本
prediction = test_text_function(test_text, bert_model, tokenizer, xgb_model, device)

# 輸出預測結果
if prediction == 0:
    print("這段文字是真實的。")
else:
    print("這段文字是假的。")










"""計算時間"""
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import time  # 引入 time 模組

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

# 計算程式執行時間
start_time = time.time()  # 記錄開始時間

pred_label, embedding = predict(test_text)

end_time = time.time()  # 記錄結束時間
elapsed_time = end_time - start_time  # 計算耗時

# **輸出結果**
print("預測結果:", "假新聞" if pred_label == 1 else "真新聞")
print(f"程式執行時間: {elapsed_time:.4f} 秒")


























import torch
from transformers import BertTokenizer, BertModel

# 載入中文 BERT 模型
model_name = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 獲取詞彙表
vocab = tokenizer.get_vocab()

# 顯示詞彙表中的一些詞
for idx, (word, idx_value) in enumerate(vocab.items()):
    if 20000 <idx < 20200:  # 只顯示前20個詞
        print(f"Word: {word}, ID: {idx_value}")
# 檢查一個例子詞語如何被分解
word = "背包"
tokens = tokenizer.tokenize(word)

# 顯示分解結果
print(f"詞語: {word}")
print(f"分解的 tokens: {tokens}")

# 如果你想將整個詞彙表儲存為檔案，可以使用以下方式：
with open("chinese_bert_wwm_vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")

print("詞彙表已經儲存到 chinese_bert_wwm_vocab.txt")
# 定義一段中文文本
text = "自然語言處理是一個有趣的領域"

# 將文本進行 tokenization，將詞拆分為子詞
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))  # 查看模型如何將文本分解為子詞
print(f"Tokenized words: {tokens}")

# 將文本轉換為模型輸入格式
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 獲取 BERT 模型的輸出
with torch.no_grad():
    outputs = model(**inputs)

# BERT 模型的輸出是每個 token 的隱藏狀態
# 我們使用最後一層隱藏狀態作為每個 token 的向量
last_hidden_states = outputs.last_hidden_state

# 輸出每個 token 的向量
print("Token-level embeddings (shape: [batch_size, sequence_length, hidden_size]):")
print(last_hidden_states.shape)

# 假設你想要獲得詞級向量，你可以將對應子詞的向量進行平均或選擇
word_embeddings = []
for i, token in enumerate(tokens):
    word_embeddings.append(last_hidden_states[0, i, :].numpy())

# 顯示每個詞對應的向量
for i, token in enumerate(tokens):
    print(f"Token: {token}, Embedding: {word_embeddings[i]}")

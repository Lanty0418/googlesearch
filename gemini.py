# -*- coding: utf-8 -*-
"""Gemini.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1emh21TXKdGJ6pLLC90_kR81CVTMT3Vsp
"""

!pip install -q -U google-generativeai

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

client = genai.Client(api_key='AIzaSyA_0T5q_GV--PVEDX2Hu36LyNqsvMH9elU')
model_id = "gemini-2.0-flash"

# 定義 Google 搜尋工具
google_search_tool = Tool(
    google_search=GoogleSearch()
)


# 設定要檢查的新聞
news_text = "初瓦韓式料理全台三分店明起歇業"

# 提示 Gemini 進行事實查核
prompt = f"""
請幫我分析以下新聞內容的真假嗎?：
"{news_text}"

"""

# 產生內容
response = client.models.generate_content(
    model=model_id,
    contents=prompt,
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

# 取得回應內容
generated_text = response.candidates[0].content
grounding_metadata = response.candidates[0].grounding_metadata



# 顯示內容與可信度
print(f"生成內容: {generated_text}")

# 輸出 grounding_metadata 資料
print(response.candidates[0].grounding_metadata)
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 16:09:35 2025

@author: lanty
"""
"""簡略的搜尋"""
import requests
import json
API_KEY = "AIzaSyBKUnW44xp7n5R2uZF-cLE5qQNVFNm9x5E"  # 申請 API Key
CX = "3297253fb8ddd402f"  # 自訂搜尋引擎 ID

query = "緬甸大地震"
url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}"

response = requests.get(url)
data = response.json()

with open('search_results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    
for item in data.get("items", []):
    print(f"標題: {item['title']}")
    print(f"網址: {item['link']}")
    print(f"摘要: {item['snippet']}\n")
    

    
"""+beautifulSoup (爬詳細新聞的內容)"""    
import requests
import json
from bs4 import BeautifulSoup

# Google 搜尋 API 設定
API_KEY = "AIzaSyBKUnW44xp7n5R2uZF-cLE5qQNVFNm9x5E"  # 申請 API Key
CX = "3297253fb8ddd402f"  # 自訂搜尋引擎 ID
query = "緬甸大地震"

# Google 搜尋 API 請求
url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}"
response = requests.get(url)
data = response.json()

# 儲存搜尋結果
with open('search_results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 解析 API 回應
if "items" in data and len(data["items"]) > 0:
    top_result = data["items"][0]  
    title = top_result['title']
    link = top_result['link']
    snippet = top_result['snippet']

    print(f"\n 最高關聯的新聞 \n")
    print(f"標題: {title}")
    print(f"網址: {link}")
    print(f"摘要: {snippet}\n")

    # 嘗試爬取完整新聞內容
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        news_response = requests.get(link, headers=headers, timeout=10)
        news_response.raise_for_status()  # 檢查請求是否成功

        soup = BeautifulSoup(news_response.text, 'html.parser')

        # 嘗試找出新聞正文（根據不同網站的 HTML 結構可能需要調整）
        article_content = []
        for tag in soup.find_all(['p', 'div']):  # 可能在 <p> 或 <div> 標籤內
            text = tag.get_text(strip=True)
            if len(text) > 50:  # 過濾掉過短的文本
                article_content.append(text)

        if article_content:
            full_content = "\n".join(article_content[:10])  # 取前 10 段
            print(f" **完整新聞內容 (前10段)** \n{full_content}\n")
        else:
            print(" 未能成功擷取完整新聞內容，可能需要調整爬取策略。")

    except requests.exceptions.RequestException as e:
        print(f" 爬取新聞內容失敗: {e}")

else:
    print(" 沒有找到相關新聞")



"""selenium結合+beautifulSoup"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# 設定 Selenium 瀏覽器
chrome_options = Options()
# chrome_options.add_argument("--headless")  # 如果需要無頭模式，可以取消註解
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1280x720")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# 啟動瀏覽器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

try:
    # Google 搜尋
    driver.get("https://www.google.com/")
    time.sleep(2)  # 等待頁面載入

    search_box = driver.find_element(By.NAME, "q")  # 找到搜尋框
    search_box.send_keys("金秀賢")  # 輸入關鍵字
    search_box.send_keys(Keys.RETURN)  # 按下 Enter
    time.sleep(3)  # 等待搜尋結果載入

    #  抓取第一個搜尋結果
    search_results = driver.find_elements(By.CSS_SELECTOR, "div.tF2Cxc a")  # Google 搜尋結果的選擇器
    links = [result.get_attribute("href") for result in search_results if "news.google.com" not in result.get_attribute("href")]

    if not links:
        print(" 沒有找到新聞連結")
        driver.quit()
        exit()

    first_news_url = links[0]  # 取第一個新聞網址
    print(f"\n 進入新聞頁面：{first_news_url}")
    driver.get(first_news_url)  # 進入新聞頁面
    time.sleep(5)  # 等待新聞載入

    #獲取頁面的 HTML 並使用 BeautifulSoup 解析
    page_html = driver.page_source  # 獲取網頁 HTML
    soup = BeautifulSoup(page_html, 'html.parser')  # 使用 BeautifulSoup 解析 HTML

    # 解析標題
    title = soup.find('h1')  # 假設標題在 <h1> 標籤內
    if title:
        print(f"\n標題: {title.text.strip()}")

    # 解析內文
    article_paragraphs = soup.find_all('p')  # 找到所有 <p> 標籤
    article_texts = [p.text.strip() for p in article_paragraphs if len(p.text.strip()) > 50]  # 篩選出長的段落

    # 顯示新聞內容
    if article_texts:
        print("\n完整新聞內容 ")
        print("\n".join(article_texts[:10]))  # 只顯示前 10 段
    else:
        print(" 未能擷取完整新聞內容")

except Exception as e:
    print(f" 發生錯誤: {e}")

finally:
    driver.quit()  # 關閉瀏覽器

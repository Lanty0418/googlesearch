from llama_cpp import Llama
import time

# 設定本地 GGUF 模型路徑
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# 初始化 Llama（設定 CPU 運行）
llm = Llama(model_path=MODEL_PATH, n_threads=8, n_batch=512)

# 設定context內容
context = """緬甸芮氏規模8.2強震災後第三日，3月31日緬甸消防局啟動空拍機，穿越第二大城瓦城，顯示倒塌後的建築層層疊疊，城市滿目瘡痍，足見地震之威力。入夜以後，街上仍然都是災民，無家可歸或者有家不敢回；救援單位則擔心，炎熱天氣下，飲用水與醫療量能都不足。"""
  
# 設定查詢問題
query_sentence = "緬甸發生9.0 地震嗎"

# 生成推理的prompt
prompt = f"""
以下是與查詢問題相關的內容：
查詢問題: {query_sentence}
新聞內容:
{context}

根據新聞內容，請回答查詢問題：{query_sentence}，並提供你的推理過程。
"""

# 使用 Llama 進行推理
start_time = time.time()
output = llm(prompt, max_tokens=512)
end_time = time.time()

# 顯示推理結果
print(f"推理耗時: {end_time - start_time:.4f} 秒")
print("\n=== AI 推理結果 ===")
print(output)

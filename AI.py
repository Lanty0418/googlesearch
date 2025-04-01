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
output = llm(prompt)
end_time = time.time()

# 顯示推理結果
print(f"推理耗時: {end_time - start_time:.4f} 秒")
print("\n=== AI 推理結果 ===")
print(output)



"""分批處理"""
from llama_cpp import Llama
import time

# 設定本地 GGUF 模型路徑
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# 初始化 Llama（設定較大的上下文長度）
llm = Llama(model_path=MODEL_PATH, n_threads=8, n_batch=512, n_ctx=4096)

# 設定新聞內容（較長）
context = """緬甸芮氏規模8.2強震災後第三日，3月31日緬甸消防局啟動空拍機，穿越第二大城瓦城，顯示倒塌後的建築層層疊疊，城市滿目瘡痍，足見地震之威力。入夜以後，街上仍然都是災民，無家可歸或者有家不敢回；救援單位則擔心，炎熱天氣下，飲用水與醫療量能都不足。
根據CNN報導，緬甸至少有2056人死亡，雖傳出有人獲救的好消息，但黃金救援72小時逐漸流逝，專家表示真正的罹難者人數，恐在災後數周才能確認，美國估計恐將破萬。緬甸處於地震帶，但過去地震較多發生在人口稀少地區，而不是像此次重擊大城市，且地震深度僅10公里，某些地區甚至「一秒移動五公尺」，因此造成嚴重損害。"""

# 設定查詢問題
query_sentence = "緬甸地震深度50公里嗎"

# 設定 Sliding Window 參數
MAX_LENGTH = 512   # 每次處理的最大 token 數
OVERLAP = 128      # 窗口重疊大小（避免上下文丟失）

def split_text(text, max_length, overlap):
    """將長文本拆成多個重疊的窗口"""
    words = text.split()  # 以空格拆分成詞
    step = max_length - overlap  # 每次移動的步長
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), step)]

# 拆分長新聞內容
chunks = split_text(context, MAX_LENGTH, OVERLAP)

# 依序處理每個窗口
responses = []
start_time = time.time()

for i, chunk in enumerate(chunks):
    print(f"\n=== 正在處理第 {i+1}/{len(chunks)} 個段落 ===")

    prompt = f"""
    查詢問題: {query_sentence}
    新聞內容:
    {chunk}

    根據以上內容，請回答：{query_sentence}，並提供你的推理過程。
    """
    
    # 使用 Llama 進行推理
    output = llm(prompt, max_tokens=256)
    
    # 取得模型回應的內容
    response_text = output["choices"][0]["text"]
    
    # 儲存每個窗口的回應
    responses.append(response_text)

end_time = time.time()

# 將所有回應合併
final_response = "\n".join(responses)

# 顯示最終結果
print(f"\n=== 全部處理完成，耗時: {end_time - start_time:.4f} 秒 ===")
print("\n=== AI 綜合推理結果 ===")
print(final_response)



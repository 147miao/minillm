import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 加载 .env 里的 Key

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "你好，用一句话介绍你自己。"}],
)
print(resp.choices[0].message.content)
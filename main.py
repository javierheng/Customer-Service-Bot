import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#学习template
# prompt = PromptTemplate.from_template(
#     "今天{city}的天气怎么样？"
# )

# 使用 DeepSeek（兼容 OpenAI 格式）
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

#创建agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

#运行代理
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
)
print(result["messages"][-1].content)

# output_parser = StrOutputParser()
# chain = prompt | model | output_parser

# result = chain.invoke({"city": "北京"})
# print(result)


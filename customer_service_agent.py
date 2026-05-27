import os                                                 # 用于读取环境变量
from dotenv import load_dotenv                            # 用于加载 .env 文件中的环境变量
import gradio as gr                                       # 用于创建前端界面
from langchain_openai import ChatOpenAI                   #用于导入可以通过OpenAI API进行对话的类
from langchain_core.prompts import ChatPromptTemplate     #用于创建聊天提示模版
from langchain_core.output_parsers import StrOutputParser #用于解析模型输出为字符串


load_dotenv() 

def get_ai_response(user_message: str, chat_history: list) -> str:
    """
    调用大模型，生成客服回复内容
    - user_message: 当前用户输入
    - chat_history: 过往对话历史（可用于上下文）
    """

    customer_service_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You area a Customer Service Agent for an online store.

            Your primary role is to assist customers with their inquiries:
            - Answer questions about products, orders, and policies.
            - Provide clear and concise information.
            - Be polite and professional at all times.
            - If you don't know the answer, say you will find out and get back to them.

            products prices:
            1. potato chips: $2.99
            2. chocolate bar: $1.49
            3. soda can: $0.99
            4. sandwich: $4.99
            5. coffee: $2.49

            When responding:
            - Always address the customer's specific question.
            - Provide accurate information based on the products and policies.
            """.strip()
        ),
        ("human","{user_message}")
    ])

    model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
    output_parser = StrOutputParser()
    chain = customer_service_prompt | model | output_parser
    result = chain.invoke({"user_message": user_message})
    return result

def chat_handler(message: str, history: list) -> str:
    """
    Gradio ChatInterface 的回调函数
    负责：
    1. 接收用户输入
    2. 调用 get_ai_response 获取 AI 回复
    3. 返回 AI 回复以更新界面
    """
    ai_reply = get_ai_response(message, history)
    return ai_reply

#使用 Gradio 创建聊天界面
chat_ui = gr.ChatInterface(
    fn=chat_handler,
    title="Customer Service Chatbot",
    description="请输入您的问题，客服机器人将为您提供帮助！"
)

if __name__ == "__main__":
    chat_ui.launch(share=True) # share=True 会生成一个公网访问链接，方便测试和分享
import os                                                 # 用于读取环境变量
from dotenv import load_dotenv                            # 用于加载 .env 文件中的环境变量
import gradio as gr                                       # 用于创建前端界面
from langchain_openai import ChatOpenAI                   #用于导入可以通过OpenAI API进行对话的类
from langchain_core.prompts import ChatPromptTemplate     #用于创建聊天提示模版
from langchain_core.output_parsers import StrOutputParser #用于解析模型输出为字符串
from langchain_core.prompts import MessagesPlaceholder    #用于在提示模版中占位对话历史
from langchain_community.chat_message_histories import ChatMessageHistory #用于存储对话历史
from langchain_core.runnables.history import RunnableWithMessageHistory #用于创建一个可以处理对话历史的可运行对象
import whisper
import edge_tts
import time


load_dotenv()

#加载 Whisper 模型，"turbo" 是一个较小的模型，适合实时转录
asr_model = whisper.load_model("turbo")

#存储不同用户的记忆
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

customer_service_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a Customer Service Agent for an online store.

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
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{user_message}")
])

model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
output_parser = StrOutputParser()

#使用链式调用将提示模版、模型和输出解析器连接起来
chain = customer_service_prompt | model | output_parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_message",
    history_messages_key="chat_history",
)

# 把用户语音转成文本
def speech_to_text(audio_path: str) -> str:
    transcribed = asr_model.transcribe(audio_path)
    return transcribed["text"]

# 把文本转成语音
def text_to_speech(text: str) -> str:
    print(f"[TTS] Processing: {text}")
    audio_path = f"./output_{int(time.time())}.mp3"
    communicate = edge_tts.Communicate(text, "en-GB-SoniaNeural")
    with open(audio_path, "wb") as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
    return audio_path


def get_ai_response(user_message: str, session_id: str) -> str:
    """
    调用大模型，生成客服回复内容
    - user_message: 当前用户输入
    - chat_history: 过往对话历史（可用于上下文）
    """

    partial_answer = ""

    for chunk in chain_with_history.stream(
        {"user_message": user_message},
        config={"configurable": {"session_id": session_id}}
    ):
        if chunk:
            partial_answer += chunk
            yield partial_answer

def process_voice_and_stream(audio_path: str, history: list):
    user_text = speech_to_text(audio_path)
    if not user_text:
        yield history, None
        return
    
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ""})
    yield history, None

    session_id = "user_001"

    full_response = ""
    for partial in get_ai_response(user_text, session_id):
        full_response = partial
        history[-1]["content"] = full_response
        #实时更新前端显示
        yield history, None

    audio_reply = text_to_speech(full_response)
    yield history, audio_reply

with gr.Blocks(theme=gr.themes.Soft()) as chat_ui:
    gr.Markdown("## 🛍️ Welcome to Our Customer Service Chatbot!")
    gr.Markdown("Ask me anything about our products, orders, or policies. I'm here to help!")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Speak Your Question"
            )
            audio_output = gr.Audio(label="🔊 AI Response", autoplay=True)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话记录")
            clear_btn =gr.Button("清空对话")

    audio_input.stop_recording(
        fn=process_voice_and_stream,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_output]
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    chat_ui.launch(share=True) # share=True 会生成一个公网访问链接，方便测试和分享
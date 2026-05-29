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
from typing import Mapping, Any
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_openai.chat_models import base
from typing import cast

# 当前langchain_openai的版本会把推理过程的文本丢掉，我们先用猴子补丁的方式让推理文本能够传递下来
_original_create_chunk = base._convert_delta_to_message_chunk

def _patched_convert_delta_to_message_chunk(
        _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:

    message_chunk = _original_create_chunk(_dict, default_class)

    try:
        role = cast(str, _dict.get("role"))
        additional_kwargs: dict = {}
        if _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
        if role == "assistant" or default_class == AIMessageChunk:
            message_chunk.additional_kwargs = additional_kwargs
    except Exception:
        pass

    return message_chunk

base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk


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

# 根据是否开启深度思考来选择模型配置
def get_model(deep_thinking: bool):
    if deep_thinking:
        return ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            extra_body={"thinking": {"type": "enabled"}}
        )
    else:
        return ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )

# 构建带历史记录的链
def build_chain(deep_thinking: bool):
    model = get_model(deep_thinking)
    chain = customer_service_prompt | model

    return RunnableWithMessageHistory(
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


def get_ai_response(user_message: str, session_id: str, deep_thinking: bool):
    """
    调用大模型，生成客服回复内容
    - user_message: 当前用户输入
    - deep_thinking: 是否开启深度思考模式
    yield: {"thinking": str, "answer": str}
    """
    chain_with_history = build_chain(deep_thinking)

    thinking_buffer = ""
    answer_buffer = ""

    for chunk in chain_with_history.stream(
        {"user_message": user_message},
        config={"configurable": {"session_id": session_id}}
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue

        # 获取推理过程（monkey patch 已将 reasoning_content 注入 additional_kwargs）
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            thinking_buffer += reasoning

        # 获取最终回复内容
        if chunk.content:
            answer_buffer += chunk.content

        yield {"thinking": thinking_buffer, "answer": answer_buffer}


def stream_ai_response(user_text: str, history: list, deep_thinking: bool):
    """核心流式响应逻辑，语音和文字输入共用"""
    if not user_text:
        yield history, gr.update(), None
        return

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ""})
    yield history, gr.update(value=""), None

    session_id = "user_001"

    thinking_md = ""
    for chunk in get_ai_response(user_text, session_id, deep_thinking):
        thinking_text = chunk["thinking"]
        answer_text = chunk["answer"]

        history[-1]["content"] = answer_text

        if thinking_text:
            thinking_md = (
                "### 🧠 思考过程\n\n"
                + thinking_text.replace("\n", "\n> ")
            )

        yield history, gr.update(value=thinking_md), None

    final_answer = history[-1]["content"]
    audio_reply = text_to_speech(final_answer)
    yield history, gr.update(value=thinking_md), audio_reply


def process_voice_and_stream(audio_path: str, history: list, deep_thinking: bool):
    """语音输入：先转文字，再走流式响应"""
    user_text = speech_to_text(audio_path)
    for output in stream_ai_response(user_text, history, deep_thinking):
        yield output


def process_text_and_stream(user_text: str, history: list, deep_thinking: bool):
    """文字输入：直接走流式响应"""
    for output in stream_ai_response(user_text, history, deep_thinking):
        yield output


with gr.Blocks(theme=gr.themes.Soft()) as chat_ui:
    gr.Markdown("## 🛍️ Welcome to Our Customer Service Chatbot!")
    gr.Markdown("Ask me anything about our products, orders, or policies. I'm here to help!")

    # 深度思考开关
    thinking_toggle = gr.Checkbox(
        label="🧠 Deep Thinking",
        value=False,
        info="Enable step-by-step reasoning before the model answers"
    )

    # 思考过程展示面板
    thinking_display = gr.Markdown("")

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
            text_input = gr.Textbox(
                label="💬 打字输入",
                placeholder="在这里输入你的问题，按 Enter 发送..."
            )
            clear_btn = gr.Button("清空对话")

    audio_input.stop_recording(
        fn=process_voice_and_stream,
        inputs=[audio_input, chatbot, thinking_toggle],
        outputs=[chatbot, thinking_display, audio_output]
    )

    text_input.submit(
        fn=process_text_and_stream,
        inputs=[text_input, chatbot, thinking_toggle],
        outputs=[chatbot, thinking_display, audio_output]
    )

    clear_btn.click(
        lambda: ([], "", None, ""),
        None,
        [chatbot, thinking_display, audio_output, text_input]
    )

if __name__ == "__main__":
    chat_ui.launch(share=True) # share=True 会生成一个公网访问链接，方便测试和分享

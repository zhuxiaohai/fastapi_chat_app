import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

base_url = os.getenv("DOUBAO_ENDPOINT")
deepseek_r1 = os.getenv("DEEPSEEK_R1")
api_key = os.getenv("DOUBAO_KEY")
client = AsyncOpenAI(
    api_key = api_key,
    base_url = base_url,
)
model = {"deepseek_r1": deepseek_r1}

app = FastAPI()

# 定义请求和响应的 Pydantic 模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 100
    stream: bool = False

# 模型列表接口
@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "deepseek_r1",
                "object": "model",
                "owned_by": "openai",
                "permission": []
            }
        ],
        "object": "list"
    }

# 流式生成器
async def stream_openai_response(chat_request: ChatRequest):
    try:
        # 调用 OpenAI 的聊天接口，启用流式模式
        response = await client.chat.completions.create(
            model=model[chat_request.model],
            messages=[{"role": msg.role, "content": msg.content} for msg in chat_request.messages],
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            stream=True  # 启用流式返回
        )

        # 逐步读取 OpenAI 的流式响应
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.reasoning_content
            if delta:
                yield f"data: {delta}\n\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# 聊天接口
@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatRequest):
    logging.debug(f"stream: {chat_request.stream}")
    # 如果启用了流式返回
    if chat_request.stream:
        return StreamingResponse(
            stream_openai_response(chat_request),
            media_type="text/event-stream"
        )

    # 非流式返回
    try:
        response = await client.chat.completions.create(
            model=model[chat_request.model],
            messages=[{"role": msg.role, "content": msg.content} for msg in chat_request.messages],
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens
        )
        logging.debug(f"response: {response}")
    
        # 返回非流式的 OpenAI 格式响应
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
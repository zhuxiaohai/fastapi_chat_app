#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
import os
from typing import Annotated
from typing_extensions import TypedDict
from fastapi import WebSocket
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import trim_messages, AIMessage, HumanMessage
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()


base_url = os.getenv("DOUBAO_ENDPOINT")
deepseek_r1 = os.getenv("DEEPSEEK_R1")
api_key = os.getenv("DOUBAO_KEY")
model = AsyncOpenAI(
    api_key = api_key,
    base_url = base_url,
)


# Define trimmer
# count each message as 1 "token" (token_counter=len) and keep only the last two messages
trimmer = trim_messages(strategy="last", max_tokens=4, token_counter=len)


async def stream_tokens(messages: list[dict]):
    response = await model.chat.completions.create(
        messages=messages, model=deepseek_r1, stream=True
    )
    role = None
    first_reasoning = True
    end_reasoning = False
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.role is not None:
            role = delta.role
        if hasattr(delta, "reasoning_content"):
            if first_reasoning:
                reasoning_content = "<Think\n" + delta.reasoning_content
                first_reasoning = False
            else:
                reasoning_content = delta.reasoning_content
        else:
            if delta.content and (not end_reasoning):
                reasoning_content = "Think>\n"
                end_reasoning = True
            else:
                reasoning_content = None
        yield {"role": role, "content": delta.content, "reasoning_content": reasoning_content}

            
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    cot: str


workflow = StateGraph(state_schema=State)


role_map = {"human": "user", "ai": "assistant"} 


# Define the function that calls the model
async def call_model(state: State, config, writer):
    trimmed_messages = trimmer.invoke(state["messages"])
    messages = [{"role": role_map[message.type], "content": message.content} for message in trimmed_messages]
    content = ""
    cot = ""
    async for msg_chunk in stream_tokens(
            messages
        ):
        content += msg_chunk["content"]
        if msg_chunk["reasoning_content"] is not None:
            cot += msg_chunk["reasoning_content"]
        metadata = {**config["metadata"]}
        chunk_to_stream = (msg_chunk, metadata)
        writer(chunk_to_stream)
    
    return {"messages": AIMessage(content=content), "cot": cot}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")


# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_input = await websocket.receive_text()

    try:
        async for chunk, metadata in app.astream(
            {"messages": [HumanMessage(content=user_input)]},
		    config={"configurable": {"thread_id": "1"}},
		    stream_mode="custom"
	    ):
            model_output = chunk["reasoning_content"] if chunk["reasoning_content"] is not None else chunk["content"]
            await websocket.send_text(model_output)
    except Exception as e:
        await websocket.send_text(f"Error: {e}")
    finally:
        await websocket.close()

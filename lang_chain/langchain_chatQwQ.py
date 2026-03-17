"""Qwen agent entrypoint."""

import os
import sys
import asyncio
import json

from pathlib import Path
from langchain_qwq import ChatQwQ
from app import load_env
from langchain_core.tools import tool
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

def main() -> None:
    load_env()

    # 1.基础用法
    model = ChatQwQ(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("QWEN_MODEL"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        temperature=float(os.getenv("QWEN_TEMPERATURE")),
        max_tokens=int(os.getenv("QWEN_MAX_TOKENS")),
        timeout=int(os.getenv("QWEN_TIMEOUT")),
    )

    response = model.invoke("please introduce yourself in Chinese, who are you. and tell how about the weather today?")
    print(response.content)

    # 2.通过 additional_kwargs 获取推理内容：需要模型支持才有返回值
    reasoning = response.additional_kwargs.get("reasoning_content", "")
    print(reasoning)

    # 3.同步流媒体
    is_first = True
    is_end = True

    for msg in model.stream("Hello, what's your name?"):
        if hasattr(msg, "additional_kwargs") and "reasoning_content" in msg.additional_kwargs:
            if is_first:
                print("Sync Starting to think...")
                is_first = False
            print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
        elif hasattr(msg, "content") and msg.content:
            if is_end:
                print("\n Sync Thinking ended")
                is_end = False
            print(msg.content, end="", flush=True)

    # 4.异步流媒体
    asyncio.run(run_async_stream(model, "Hello, what's your name?"))

    # 5.内容块：原始结构化数据
    print(response.content_blocks)

    # 6.工具调用：需要模型支持工具调用功能才有返回值
    bind_model = model.bind_tools([get_weather])
    bind_model_response = bind_model.invoke("获取上海的天气")

    # 打印模型工具调用列表，包含工具调用的名称、参数等信息，需要模型支持工具调用功能才有返回值
    print(bind_model_response.tool_calls)

    # QwQ 的工具调用，需要手动调用
    for call in bind_model_response.tool_calls:
        tool_name = call["name"]
        args = call["args"]

        if tool_name == "get_weather":
            result = get_weather.run(args["city"])
            print("手动执行工具调用结果:", result)
    # QwQ 不会自动将工具调用结果返回给模型，所以需要手动将工具调用结果传回模型进行后续推理
    print("工具调用结果：", bind_model_response.content)
    
    # 7. 结构化输出
    struct_model = model.with_structured_output(UserInfo, method="json_mode")
    struct_response = struct_model.invoke("Hello, I'm Nakajima, 36 years old, living in Shanghai.")
    print("结构化输出结果：", struct_response)
    print("结构化输出结果：", struct_response.name)

    # 8. 函数调用模式
    struct_model1 = model.with_structured_output(UserInfo, method="function_calling")
    struct_response1 = struct_model1.invoke("My name is Alice and I'm 30")
    print("函数调用模式结构化输出结果：", struct_response1)

    # 9. 集成langchainAgents
    agent = create_agent(model, [get_weather])
    print(agent.invoke({"messages":[HumanMessage("What is the weather like in 上海?")]}))


# attention: 异步流媒体需要在支持异步的环境中运行
# async for 只能在async函数中使用，所以需要定义一个async函数来处理异步流媒体
async def run_async_stream (model, param, is_first=True, is_end=True):
    stream = model.astream(param)
    async for msg in stream:
        if hasattr(msg, 'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
            if is_first:
                print("Async Starting to think...")
                is_first = False
            print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
        elif hasattr(msg, 'content') and msg.content:
            if is_end:
                print("\n Async Thinking ended")
                is_end = False
            print(msg.content, end="", flush=True)

@tool
def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"The weather in {city} is sunny."

class UserInfo(BaseModel):
        name: str
        age: int
        city: str

def handle_user_info(name: str, age: int, city: str = None):
    return UserInfo(name=name, age=age, city=city)

if __name__ == "__main__":
    main()

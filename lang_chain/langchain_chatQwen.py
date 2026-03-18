import os

from app import load_env
from langchain_qwq import ChatQwen
from langchain_core.tools import tool
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


def main() -> None:
    load_env()

    # 1.基础用法
    model = ChatQwen(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("QWEN_MODEL"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        temperature=float(os.getenv("QWEN_TEMPERATURE")),
        max_tokens=int(os.getenv("QWEN_MAX_TOKENS")),
        timeout=int(os.getenv("QWEN_TIMEOUT")),
        # True 开启推理内容返回，默认为 False
        enable_thinking=False,
        # 控制思维长度，默认为 20，单位为 token
        thinking_budget=20,
    )

    response = model.invoke("please simple introduce yourself in Chinese.")
    print("自我介绍：", response.content)

    # 2.通过 additional_kwargs 获取推理内容：需要模型支持才有返回值
    reasoning = response.additional_kwargs.get("reasoning_content", "")
    print("推理内容：", reasoning)

    # 3. 内容块：原始结构化数据
    print("内容块：", response.content_blocks)

    # 4. 工具调用：需要模型支持工具调用功能才有返回值
    # 在agent中才会自动调用工具，这里直接调用需要设置 parallel_tool_calls=True 来开启工具并行调用功能，否则会等待第一个工具调用完成才会继续后续工具调用
    bind_model = model.bind_tools([get_this_weather])
    bind_model_response = bind_model.invoke("获取上海的天气", parallel_tool_calls=True)
    print("工具调用列表：", bind_model_response.tool_calls)
    print("工具调用结果：", bind_model_response.content)

    # 5. 结构化输出
    struct_model = model.with_structured_output(UserInfo, method="json_mode")
    struct_response = struct_model.invoke("Hello, I'm Nakajima, 36 years old, living in Shanghai.")
    print("结构化输出结果：", struct_response)
    print("结构化输出结果：", struct_response.name)

    # 6. 集成langchainAgents
    agent = create_agent(model, [get_this_weather])
    result = agent.invoke({"messages":[HumanMessage("What is the weather like in 上海?")]})
    print(result)
    # 取最后一条 AI 消息内容并打印
    msgs = result.get("messages", [])
    if msgs:
        print("Agent消息：", msgs[-1].content)
    else:
        print(result)

@tool
def get_this_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"The weather in {city} is sunny."

class UserInfo(BaseModel):
        name: str
        age: int
        city: str


if __name__ == "__main__":
    main()
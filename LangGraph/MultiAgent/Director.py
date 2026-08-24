import asyncio
from operator import add
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from config.load_key import load_key
from langchain_community.chat_models import ChatTongyi

nodes = ["supervisor", "travel", "joke", "couplet", "other"]

llm = ChatTongyi(
    model="qwen-plus-2025-07-28",
    api_key=load_key("BAILIAN_API_KEY"),
)
class State(TypedDict):
    messages:Annotated[list[AnyMessage],add]
    type:str


def other_node(state:State):
    print(">>> other_node")
    writer = get_stream_writer()
    writer({"node": ">>> other_node"})
    return {"messages":[HumanMessage(content="我暂时无法回答这个问题")],"type":"other"}

def supervisor_node(state:State):
    print(">>> supervisor_node")
    writer = get_stream_writer()
    writer({"node": ">>> supervisor_node"})
    # 根据用户的问题，对问题进行分类，分类结果保存到type当中
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他agent执行，
                如果用户的问题是和旅游路线规划相关的，那就返回 travel。
                如果用户的问题是和笑话相关的，那就返回 joke。
                如果用户的问题是和对联相关的，那就返回 couplet。
                如果用户的问题不属于以上三种情况，那就返回 other。
                除了上述几个选项外，不返回任何其他的内容。
                """
    # 取最后一条用户消息文本
    
    prompts = [
        {"role":"system", "content":prompt},
        {"role":"user", "content":state["messages"][0]}
    ]
    # 如果已经有type属性了，表示问题已经交由其他节点处理完成了，就可以直接返回
    if "type" in state:
        writer({"supervisor_step": f"已获得{state['type']}智能体处理结果"})
        return {"type":END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content
        writer({"supervisor_step": f"问题分类结果：{typeRes}"})
        if typeRes in nodes:
            return {"type":typeRes}
        else:
            return {"type":"other"}


def travel_node(state:State):
    print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": ">>> travel_node"})

    System_Prompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游规划。请用中文回答，并返回一个不超过100字的回答。"
    prompts = [
        {"role": "system", "content": System_Prompt},
        {"role": "user", "content": state["messages"][0]}
    ]

    # 高德地图的MCP配置信息
    client = MultiServerMCPClient(
        {
            # sse接入方式
            # "amap-amap-sse": {
            #    "url": "https://mcp.amap.com/sse?key=451ad40d0e39453600f2a305e31eabe4",
            #    "transport": "streamable_http"
            # },
            # stdio接入方式
            "amap-maps": {
                "command": "npx",
                "args": [
                    "-y",
                    "@amap/amap-maps-mcp-server"
                ],
                "env": {
                    "AMAP_MAPS_API_KEY": "451ad40d0e39453600f2a305e31eabe4"
                },
                "transport": "stdio"
            }
        }
    )
    tools = asyncio.run(client.get_tools())
    agent = create_react_agent(
        model=llm,
        tools=tools
    )
    response = agent.invoke({"messages":prompts})
    writer({"travel_result":response["messages"][-1].content})
    return {"messages":[HumanMessage(content=response["messages"][-1].content)],"type":"travel"}

def joke_node(state:State):
    print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": ">>> joke_node"})
    System_Prompt = "你是一个笑话大师，根据用户的问题写一个不超过100字的笑话。"
    prompts = [
        {"role":"system", "content":System_Prompt},
        {"role":"user", "content":state["messages"][0]}
    ]
    response = llm.invoke(prompts)
    writer({"joke_result":response.content})
    return {"messages":[HumanMessage(content=response.content)],"type":"joke"}

def couplet_node(state:State):
    print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>> couplet_node"})


    return {"messages":[HumanMessage(content="couplet_node")],"type":"couplet"}


# 条件路由
def routing_func(state:State):
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "couplet":
        return "couplet_node"
    elif state["type"] == END:
        return END
    else:
        return "other_node"


# 构建图
builder = StateGraph(State)
# 创建节点，起名字
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

# 添加边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func,["travel_node", "joke_node", "couplet_node", "other_node", END])
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# 构建graph
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer = checkpointer)

# 执行任务的测试代码
if __name__ == "__main__":
    config = {
        "configurable":{
            "thread_id":"1"
        }
    }

    for chunk in graph.stream({"messages":["给我一个太原到古交的路线规划。"]}
            ,config
            ,stream_mode="custom"):
        print(chunk)

    # res = graph.invoke({"messages":["今天天气怎么样"]}
    #                    ,config
    #                    ,stream_mode="values")
    # print(res["messages"][-1].content)
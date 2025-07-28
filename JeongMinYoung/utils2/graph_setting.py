from typing import TypedDict, Literal, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from.graph_node import ChatState, ChatTurn, classify, extract_entities, handle_accounting1, handle_financial1, handle_business1, handle_hybrid1, elief, route_from_extract, route_from_classify

def graph_setting():
    # LangGraph 정의
    graph = StateGraph(ChatState)

    # 노드 등록
    graph.add_node("classify", classify)
    graph.add_node("extract", extract_entities)
    graph.add_node("accounting", handle_accounting1)
    graph.add_node("financial", handle_financial1)
    graph.add_node("business", handle_business1)
    graph.add_node("hybrid", handle_hybrid1)
    graph.add_node("simple", elief)

    # 흐름 정의
    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_from_classify,
        {
            "accounting": "extract",  # extract를 거쳐서 처리
            "finance": "extract",
            "business": "extract",
            "hybrid": "extract",
            "simple": "simple"  # simple은 바로 처리
        }
    )
    # extract 결과에 따라 분기
    graph.add_conditional_edges(
        "extract",
        route_from_extract,
        {
            "accounting": "accounting",
            "financial": "financial",
            "business": "business",
            "hybrid": "hybrid"
        }
    )

    # 종료 노드 지정
    for node in ["accounting", "financial", "business", "hybrid", "simple"]:
        graph.add_edge(node, END)

    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    return compiled_graph
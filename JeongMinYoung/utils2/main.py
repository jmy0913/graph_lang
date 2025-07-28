from .graph_setting import graph_setting

compiled_graph = graph_setting()

# 초급 전체 분기 실행 함수
def run_langraph(user_input, config_id):
    config = {"configurable": {"thread_id": config_id}}
    result = compiled_graph.invoke({
        "question": user_input
    }, config=config)

    return result['answer']
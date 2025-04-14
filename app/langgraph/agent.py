from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import NodeInterrupt
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .rag_node import retrieve_knowledge, generate_query, format_docs, should_use_rag
from .state import AgentState
from .tools import tools
from ..models import AnyArgsSchema
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Default system prompt for the financial assistant
DEFAULT_SYSTEM_PROMPT = """
Bạn là trợ lý tài chính thông minh MoneyEZ, một trợ lý AI được tạo ra để giúp người dùng quản lý tài chính cá nhân.
Nhiệm vụ của bạn:
1. Giúp người dùng theo dõi chi tiêu hàng ngày
2. Phân loại các khoản chi tiêu vào các danh mục phù hợp
3. Cung cấp thông tin và tư vấn tài chính
4. Trả lời mọi câu hỏi liên quan đến tài chính cá nhân một cách chính xác và hữu ích
5. Nếu có bất kỳ thông tin nào không rõ ràng, hãy yêu cầu người dùng cung cấp thêm thông tin.
Trả lời ngắn gọn và rõ ràng, không có markdown hay định dạng phức tạp.
"""

# Initialize the default model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def should_continue(state):
    """Determine if the agent should continue with tool execution or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    else:
        return "tools"


class FrontendTool(BaseTool):
    """Tool implementation for frontend-rendered tools."""

    def __init__(self, name: str):
        super().__init__(name=name, description="", args_schema=AnyArgsSchema)

    def _run(self, *args, **kwargs):
        raise NodeInterrupt("This is a frontend tool call")

    async def _arun(self, *args, **kwargs) -> str:
        raise NodeInterrupt("This is a frontend tool call")


def get_tool_defs(config):
    """Get tool definitions for binding to the model."""
    frontend_tools = [
        {"type": "function", "function": tool}
        for tool in config["configurable"]["frontend_tools"]
    ]
    return tools + frontend_tools


def get_tools(config):
    """Get tool instances for the tool node."""
    frontend_tools = [
        FrontendTool(tool.name) for tool in config["configurable"]["frontend_tools"]
    ]
    return tools + frontend_tools


async def call_model(state, config):
    """Call the language model with the current state and RAG context."""
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    print(f"\n[AGENT] Calling model for thread: {thread_id}")

    # Get the system prompt from config
    system = config.get("configurable", {}).get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    print(f"[AGENT] System prompt: {system}")

    # Add RAG context to system prompt if available
    if "retrieved_docs" in state and state["retrieved_docs"]:
        print(f"[AGENT] Using RAG context with {len(state['retrieved_docs'])} documents")
        rag_context = format_docs(state["retrieved_docs"])
        enhanced_system = f"{system}\n\nRelevant information from knowledge base:\n{rag_context}"
    elif "rag_context" in state and state["rag_context"]:
        print(f"[AGENT] Using RAG context with {len(state['rag_context'])} documents")
        rag_context = "\n\n".join(state["rag_context"])
        enhanced_system = f"{system}\n\nRelevant information from knowledge base:\n{rag_context}"
    else:
        print("[AGENT] No RAG context available")
        enhanced_system = system

    # Prepare messages with enhanced system prompt and ChatPromptTemplate
    messages = state.get("messages", [])
    if not messages:
        # Handle the case when messages is empty
        return {"messages": [SystemMessage(content=enhanced_system)]}
    
    # Create prompt template with system message and agent_scratchpad
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", enhanced_system),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    # Format the prompt with the chat history
    formatted_prompt = prompt.format_messages(
        chat_history=messages,
        agent_scratchpad=[]  # This can be used for agent's intermediate reasoning if needed
    )
    
    print(f"[AGENT] Total messages in context: {len(formatted_prompt)}")

    # Invoke model with tools
    print(f"[AGENT] Invoking model")
    model_with_tools = model.bind_tools(get_tool_defs(config))
    response = await model_with_tools.ainvoke(
        formatted_prompt,
        {
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )

    # Return the response to be added to the messages
    return {"messages": response}


async def run_tools(input, config, **kwargs):
    """Execute tools based on the model's response."""
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    print(f"\n[TOOLS] Running tools for thread: {thread_id}")

    tool_node = ToolNode(get_tools(config))
    response = await tool_node.ainvoke(input, config, **kwargs)

    print(f"[TOOLS] Tool response received")
    return response


# Define the simplified RAG-enabled agent workflow
workflow = StateGraph(AgentState)

# Add nodes for the RAG pipeline and agent
workflow.add_node("rag_decision", should_use_rag)
workflow.add_node("generate_query", generate_query)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("agent", call_model)
workflow.add_node("tools", run_tools)

# Set up the graph flow with conditional RAG
workflow.set_entry_point("rag_decision")

# Add conditional edges
workflow.add_conditional_edges(
    "rag_decision",
    lambda state: "use_rag" if state.get("need_rag", False) else "skip_rag",
    {
        "use_rag": "generate_query",
        "skip_rag": "agent"
    }
)

# Continue with RAG flow if needed
workflow.add_edge("generate_query", "retrieve")
workflow.add_edge("retrieve", "agent")

# Handle tools and end conditions
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Compile graph with memory checkpointer
memory = MemorySaver()
assistant_ui_graph = workflow.compile(checkpointer=memory)
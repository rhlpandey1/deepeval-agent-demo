
"""
chatbot.py
==========

Multi-turn customer-support chatbot with tool calling using Claude.

Tools:
 - get_order_status(order_id)
 - get_refund_policy(category)

chat() returns:
 - reply        : final assistant response
 - history      : updated LangChain message history
 - tools_called : list of tools invoked during this turn
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)

# ---------------------------------------------------------------------------
# In-memory data
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD-1042": {"status": "Shipped", "eta": "2026-05-13"},
    "ORD-2099": {"status": "Delivered", "eta": "2026-05-08"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing": "Clothing can be returned within 30 days with tags attached.",
    "food": "Food items are non-returnable for safety reasons.",
    "furniture": "Furniture can be returned within 30 days if unassembled.",
}

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer order by its ID."""

    order_id = order_id.upper()
    order = ORDERS.get(order_id)

    if not order:
        return f"No order found with ID {order_id}."

    return (
        f"Order {order_id} is {order['status']}. "
        f"ETA: {order['eta']}."
    )


@tool
def get_refund_policy(category: str) -> str:
    """Return the refund or return policy for a product category."""

    category = category.lower()
    policy = REFUND_POLICIES.get(category)

    if not policy:
        return f"No refund policy on file for '{category}'."

    return policy


# Register tools
TOOLS = [
    get_order_status,
    get_refund_policy,
]

TOOL_MAP = {
    tool.name: tool
    for tool in TOOLS
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a friendly and professional customer-support chatbot
for ShopEasy, an online retail store.

You have access to two tools:
- get_order_status: use whenever a customer asks about a specific order
- get_refund_policy: use whenever a customer asks about returns or refunds

Rules:
- Always use the tools when you have the information needed to call them.
- Be concise and polite.
- Never discuss topics outside of ShopEasy customer support.
- Remember everything the customer tells you in the current conversation.
"""

# ---------------------------------------------------------------------------
# Anthropic LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
)

# Bind tools to Claude
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# chat() — one user turn
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: list | None = None,
) -> tuple[str, list, list]:

    """
    Send one user message, handle tool calls, and return the final reply.

    Returns:
        reply        : final assistant text
        history      : updated LangChain message history
        tools_called : tools invoked during this turn
    """

    if history is None:
        history = []

    # Copy history so the caller's original list isn't modified
    history = list(history)

    # Append the new user message
    history.append(HumanMessage(content=user_message))

    tools_called = []

    while True:

        # Send complete conversation history to Claude
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *history,
            ]
        )

        # No tool calls: Claude has produced the final response
        if not response.tool_calls:

            reply = response.content

            # Claude content can be a list of content blocks.
            if isinstance(reply, list):
                reply = "".join(
                    block.get("text", "")
                    for block in reply
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                )

            history.append(response)

            return reply, history, tools_called

        # Append Claude's assistant message containing tool calls
        history.append(response)

        # Execute all requested tools
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            selected_tool = TOOL_MAP.get(tool_name)

            if selected_tool:
                result = selected_tool.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            tools_called.append({
                "name": tool_name,
                "args": tool_args,
                "result": result,
            })

            # Return tool output to Claude
            history.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("ShopEasy Support Chatbot (Claude + Tools) — type 'quit' to exit\n")

    history = []

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit"):
            break

        reply, history, tools = chat(user_input, history)

        if tools:
            for t in tools:
                print(
                    f"  [tool] {t['name']}({t['args']}) "
                    f"→ {t['result']}"
                )

        print(f"Bot: {reply}\n")

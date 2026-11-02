
"""
rag_agent.py
============

Customer-support RAG agent using:
- Claude Sonnet 4.6 for answer generation
- Local Hugging Face embeddings for semantic retrieval
- LangChain InMemoryVectorStore
- DeepEval CallbackHandler and tracing
"""

from dotenv import load_dotenv
load_dotenv()

from functools import lru_cache

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

POLICY_DOCS = [
    # Refund policies
    "Electronics refund policy: Electronics can be returned within 15 days of delivery, "
    "provided the item is unopened and in its original packaging. Opened electronics are "
    "non-returnable unless faulty. Refunds are processed within 5–7 business days.",

    "Clothing refund policy: Clothing items can be returned within 30 days of delivery "
    "with all original tags still attached. Items must be unworn and unwashed. "
    "Refunds are issued to the original payment method within 3–5 business days.",

    "Food refund policy: Food and perishable items are non-returnable for health and "
    "safety reasons. If your food order arrived damaged or spoiled, contact support "
    "within 24 hours and we will issue a full refund or replacement.",

    "Furniture refund policy: Furniture can be returned within 30 days if unassembled "
    "and in original packaging. Assembled furniture cannot be returned unless defective. "
    "Return shipping costs are the customer's responsibility.",

    "Jewellery refund policy: Jewellery and personalised items are non-returnable unless "
    "received in a damaged or incorrect condition. Please inspect items upon delivery.",

    # Shipping policies
    "Standard shipping policy: Standard shipping takes 5–7 business days. "
    "Orders over $50 qualify for free standard shipping. Tracking information is "
    "emailed once the order ships.",

    "Express shipping policy: Express shipping takes 1–2 business days and costs $15. "
    "Express orders placed before 2 PM are dispatched the same day.",

    "International shipping policy: International orders ship via courier and take "
    "10–21 business days. Customs duties and import taxes are the buyer's responsibility. "
    "We ship to over 50 countries.",

    # Order management
    "Order cancellation policy: Orders can be cancelled within 1 hour of placement "
    "for a full refund. After 1 hour, the order enters fulfilment and cannot be "
    "cancelled. Contact support immediately if you need to cancel.",
]


# ---------------------------------------------------------------------------
# Lazy vector-store initialization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = InMemoryVectorStore(embedding=embeddings)

    store.add_texts(POLICY_DOCS)

    return store


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------

_last_retrieved: list[str] = []


@tool
def search_policies(query: str) -> str:
    """Search the customer-support knowledge base for policy information."""

    global _last_retrieved

    vector_store = get_vector_store()

    docs = vector_store.similarity_search(query, k=3)

    chunks = [doc.page_content for doc in docs]

    _last_retrieved = chunks

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Claude agent
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[search_policies],
    system_prompt=(
        "You are a friendly customer-support agent. "
        "Use the search_policies tool to look up refund, shipping, and order policies. "
        "Answer only from the retrieved information. Keep replies concise."
    ),
)


# ---------------------------------------------------------------------------
# DeepEval callback handler
# ---------------------------------------------------------------------------

deepeval_callback = CallbackHandler()


# ---------------------------------------------------------------------------
# RAG agent entry point
# ---------------------------------------------------------------------------

def rag_support_agent(user_input: str) -> str:

    """Run the RAG agent and return the final reply."""

    global _last_retrieved

    _last_retrieved = []

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
        config={
            "callbacks": [deepeval_callback]
        },
    )

    reply = result["messages"][-1].content

    update_current_trace(
        output=reply,
        retrieval_context=_last_retrieved if _last_retrieved else None,
    )

    return reply


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    response = rag_support_agent(
        "What is the return policy for electronics?"
    )

    print("RAG Agent Response:")
    print(response)

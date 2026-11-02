
GOLDEN_DATA = [
    {
        "input": "Where is my order ORD-1042?",
        "expected_tools": ["get_order_status"],
        "expected_output": (
            "Order ORD-1042 has shipped. ETA: 2026-05-13."
        ),
    },
    {
        "input": "What is the refund policy for electronics?",
        "expected_tools": ["get_refund_policy"],
        "expected_output": (
            "Electronics can be returned within 15 days, "
            "unopened."
        ),
    },
    {
        "input": "What is the return policy for electronics?",
        "expected_tools": [],
        "expected_output": (
            "Electronics can be returned within 15 days of delivery if unopened "
            "and in original packaging. Refunds take 5–7 business days."
        ),
    },
    {
        "input": "How long does express shipping take and what does it cost?",
        "expected_tools": [],
        "expected_output": (
            "Express shipping takes 1–2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."
        ),
    },
]

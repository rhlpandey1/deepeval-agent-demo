
GOLDEN_DATA = [
    {
        "input": "Where is my order ORD-1042?",
        "expected_tools": ["get_order_status"],
        "expected_output": (
            "Order ORD-1042 has shipped and is at the "
            "Bengaluru sorting facility. ETA: 2026-05-13."
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
]

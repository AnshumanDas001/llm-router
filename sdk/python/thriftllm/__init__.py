"""ThriftLLM: calibrate your models, then route every prompt to the cheapest
one that can answer it.

    import thriftllm

    thriftllm.configure(api_key="rtr_...", base_url="https://your-thriftllm-host")

    small = thriftllm.calibrate("groq/llama-3.1-8b-instant", api_key=GROQ_KEY)
    big = thriftllm.calibrate("openai/gpt-5", api_key=OPENAI_KEY)

    router = thriftllm.Router(cheap=small, frontier=big)
    reply = router.chat("What is the capital of Australia?")
    print(reply.text, reply.tier, reply.cost)
    print(reply.explain())
"""
from thriftllm.client import (
    CalibratedModel,
    Client,
    Reply,
    Router,
    ThriftLLMError,
    calibrate,
    configure,
)

__all__ = ["CalibratedModel", "Client", "Reply", "Router", "ThriftLLMError", "calibrate", "configure"]
__version__ = "0.1.0"

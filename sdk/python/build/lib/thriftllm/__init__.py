"""ThriftLLM: calibrate your models, then route every prompt to the cheapest
one that can answer it.

    pip install "git+https://github.com/AnshumanDas001/llm-router.git#subdirectory=sdk/python"

    import os
    import thriftllm

    GROQ_KEY = os.environ["GROQ_API_KEY"]

    thriftllm.configure(api_key="rtr_...", base_url="https://your-host")
    small  = thriftllm.calibrate("groq/openai/gpt-oss-20b",  api_key=GROQ_KEY)
    medium = thriftllm.calibrate("groq/openai/gpt-oss-120b", api_key=GROQ_KEY)
    router = thriftllm.Router(cheap=small, mid=medium)
    reply  = router.chat("What is the capital of Australia?")
    print(reply.text, reply.tier, reply.cost)
    print(reply.explain())          # the same route "Why this route?" shows
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

"""Request bodies for every endpoint."""
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # ignored; the cascade always decides the tier


class AuthRequest(BaseModel):
    username: str
    password: str


class CreateChatRequest(BaseModel):
    title: str | None = None
    mode: str = "builtin"  # "builtin" (our models) or "byom" (this user's configured models)
    models: dict[str, str] = {}  # tier -> model_name, frozen onto the chat at creation
    routing_mode: str = "cascade"  # "cascade": cheapest capable tier first; "direct": skip the cheap tier


class UpdateChatRequest(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    routing_mode: str | None = None


class SendMessageRequest(BaseModel):
    content: str
    tier_api_keys: dict[str, str] = {}  # BYOM chats only; used transiently, never stored


class ClassifyRequest(BaseModel):
    content: str
    chat_id: int | None = None  # predict with this chat's own routing map


class DemoRequest(BaseModel):
    content: str
    history: list[Message] = []          # earlier turns of this demo conversation
    routing_mode: str = "cascade"


class CreateProviderRequest(BaseModel):
    name: str
    provider: str                      # litellm prefix: groq, gemini, openai, ...
    api_key: str | None = None
    store_key: bool = False            # opt in to encrypted storage; default is per-session
    api_base: str | None = None
    models: list[str] = []


class AddModelRequest(BaseModel):
    model_name: str


class CalibrateModelRequest(BaseModel):
    model_name: str
    api_key: str | None = None         # omitted when the provider has a stored key
    # API only: connect the model first if it isn't connected yet
    provider: str | None = None        # litellm prefix; inferred from "groq/..." when omitted
    api_base: str | None = None        # e.g. a local Ollama


class CreateApiKeyRequest(BaseModel):
    name: str | None = None


class ClassifyApiRequest(BaseModel):
    prompt: str
    models: dict[str, str] = {}         # tier -> model_name; omit for the built-in stack
    routing_mode: str = "cascade"


class RoutingMapRequest(BaseModel):
    models: dict[str, str] = {}         # tier -> model_name; omit for the built-in stack


class RouteRequest(BaseModel):
    messages: list[Message]
    models: dict[str, str] = {}         # tier -> model_name, from your connected models
    tier_api_keys: dict[str, str] = {}  # used transiently for this call only, never stored
    routing_mode: str = "cascade"       # or "direct" to skip the cheapest tier and its judge

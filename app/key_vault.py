"""Encryption at rest for the provider API keys a user chooses to store.

Storing third-party provider keys is opt-in per provider. The default
stays what it always was -- the key lives in the browser tab for the
session and is sent per request -- because that needs no trust in our
storage at all. This module exists only for users who'd rather trade that
for not re-entering the key after every reload.

Two deliberate properties:

* No server secret means storing is *refused*, not silently downgraded to
  plaintext. A "we encrypted it" claim we can't honor is worse than an
  error that says storage is unavailable.
* The secret lives in the environment, never in the database, so a stolen
  copy of router.db alone does not yield the keys.
"""
import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken

SECRET_ENV_VAR = "ROUTER_SECRET_KEY"


class KeyStorageUnavailable(RuntimeError):
    """Raised when key storage is requested but no server secret is set."""


def storage_available() -> bool:
    return bool(os.getenv(SECRET_ENV_VAR))


def _fernet() -> Fernet:
    secret = os.getenv(SECRET_ENV_VAR)
    if not secret:
        raise KeyStorageUnavailable(
            f"{SECRET_ENV_VAR} is not set, so provider keys cannot be stored encrypted. "
            f"Either set it, or connect the provider without saving a key."
        )
    # Fernet needs exactly 32 url-safe base64 bytes; hash whatever passphrase
    # the operator set so any length of secret works.
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(token: str) -> str | None:
    """None if the stored value can't be read back -- which happens if the
    server secret was rotated or lost. Callers treat that the same as "no
    stored key" and ask the user for it, rather than failing the request."""
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, KeyStorageUnavailable):
        return None

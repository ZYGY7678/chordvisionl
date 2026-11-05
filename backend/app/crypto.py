"""Encryption for stored third-party API keys.

The requirement is that a provider key the user types never appears in
client JavaScript, never reaches localStorage, never appears in a log
line, and is not readable from a database dump alone.

What is implemented:

  * the key is encrypted at rest with Fernet (AES-128-CBC with an
    HMAC-SHA256 authentication tag), using a key derived from the
    application secret with PBKDF2-HMAC-SHA256;
  * only the last four characters are stored in clear, so the UI can show
    the user which key is saved without being able to reconstruct it;
  * the plaintext is decrypted in memory only, at the moment a
    transcription request is made.

What this does **not** protect against, stated plainly: an attacker who
obtains both the database and the application secret can decrypt the
keys.  Protecting against that needs a real key management service (AWS
KMS, GCP KMS, Vault), and the interface here is deliberately small
enough to swap for one.
"""
from __future__ import annotations

import base64
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import get_settings

_SALT = b"chordvision/api-key-encryption/v1"
_ITERATIONS = 390_000


class DecryptionError(RuntimeError):
    """Raised when a stored key cannot be decrypted."""


def _fernet() -> Fernet:
    secret = get_settings().resolved_secret().encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=_SALT, iterations=_ITERATIONS,
    )
    return Fernet(base64.urlsafe_b64encode(kdf.derive(secret)))


def encrypt_api_key(plaintext: str) -> str:
    if not plaintext:
        raise ValueError("refusing to encrypt an empty API key")
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_api_key(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise DecryptionError(
            "The stored API key could not be decrypted. This normally means "
            "CHORDVISION_SECRET_KEY changed since the key was saved. "
            "Re-enter the key in settings."
        ) from exc


def key_hint(plaintext: str) -> str:
    """A safe-to-display fragment: the last four characters only."""
    tail = plaintext[-4:] if len(plaintext) >= 4 else ""
    return f"****{tail}"


def redact(text: str, *secrets_: Optional[str]) -> str:
    """Strip known secrets out of a string before it is logged or returned."""
    out = text
    for s in secrets_:
        if s and len(s) >= 6:
            out = out.replace(s, "***redacted***")
    return out

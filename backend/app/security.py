"""Identity and rate limiting.

Identity
--------
The specification asks for authentication only if projects are saved,
which they are.  Full accounts with passwords are more than the product
needs on day one and are a worse first experience: a person who wants to
see what the chord engine does should not have to sign up first.

So identity is a signed cookie holding a random user id.  It is created
on the first request, it is signed with the application secret so it
cannot be forged to read another person's projects, and every project
query is scoped to it.  Optional email and password columns exist in the
schema, so adding real accounts later is additive rather than a rewrite.

What this gives: projects are private to the browser that made them, and
one user cannot read another's by guessing an id.
What it does not give: recovery, or access from a second device.  Both
need real accounts.

Rate limiting
-------------
A fixed-window counter per client, held in memory.  It protects a single
instance from a runaway client, which is what it is for.  It is not a
distributed limiter; behind more than one instance this belongs in Redis
or at the load balancer.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from .config import get_settings
from .db import User, UserSettings, get_session

COOKIE_NAME = "cv_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def _sign(user_id: str) -> str:
    secret = get_settings().resolved_secret().encode("utf-8")
    mac = hmac.new(secret, user_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"{user_id}.{mac}"


def _verify(token: str) -> Optional[str]:
    if not token or "." not in token:
        return None
    user_id, _, mac = token.rpartition(".")
    expected = hmac.new(
        get_settings().resolved_secret().encode("utf-8"),
        user_id.encode("utf-8"), hashlib.sha256,
    ).hexdigest()[:32]
    if hmac.compare_digest(mac, expected):
        return user_id
    return None


def get_current_user(
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
) -> User:
    """Resolve the caller, creating an anonymous identity on first contact."""
    token = request.cookies.get(COOKIE_NAME)
    user_id = _verify(token) if token else None

    user = session.get(User, user_id) if user_id else None
    if user is None:
        user = User(display_name=None)
        session.add(user)
        session.flush()
        session.add(UserSettings(user_id=user.id))
        session.commit()
        settings = get_settings()
        response.set_cookie(
            COOKIE_NAME, _sign(user.id),
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=settings.is_production,
            path="/",
        )
    return user


class RateLimiter:
    """Fixed-window request counter, keyed by client."""

    def __init__(self):
        self._hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, key: str, bucket: str, limit: int, window_s: float) -> None:
        now = time.monotonic()
        q = self._hits[(key, bucket)]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            retry = int(window_s - (now - q[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many requests. Try again in {retry} seconds.",
                headers={"Retry-After": str(retry)},
            )
        q.append(now)

    def prune(self, max_entries: int = 10_000) -> None:
        if len(self._hits) <= max_entries:
            return
        now = time.monotonic()
        stale = [k for k, q in self._hits.items() if not q or now - q[-1] > 3600]
        for k in stale:
            del self._hits[k]


limiter = RateLimiter()


def client_key(request: Request) -> str:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        uid = _verify(token)
        if uid:
            return f"u:{uid}"
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown")
    return f"ip:{ip}"


def rate_limit(request: Request) -> None:
    settings = get_settings()
    limiter.prune()
    limiter.check(client_key(request), "general", settings.rate_limit_per_minute, 60.0)


def upload_rate_limit(request: Request) -> None:
    settings = get_settings()
    limiter.check(client_key(request), "upload", settings.upload_rate_limit_per_hour, 3600.0)

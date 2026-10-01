"""Local authentication + role-based access control for RIGAL-AI.

Single-machine console: no external identity provider. Sessions are random
in-memory tokens (not signed JWTs) — the simplest correct thing for a process
that already restarts cleanly, where "everyone re-logs-in after a restart" is
an acceptable, even slightly safer, default than persisting session state.

Roles:
  sysadmin  — full access, including User Management.
  ml_user   — everything except User Management (enforced here AND in the UI).
"""
import hashlib
import hmac
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

ROLES = {"sysadmin", "ml_user"}
PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS).hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    candidate = _hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, hash_hex)


class AuthError(Exception):
    pass


class AuthStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._sessions = {}  # token -> {"username":..., "role":...}
        self.users = self._read()
        if not self.users:
            self._bootstrap_admin()

    def _read(self):
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write(self):
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.users, indent=2))
        temp.replace(self.path)

    def _bootstrap_admin(self):
        password = secrets.token_urlsafe(12)
        salt = secrets.token_bytes(16)
        self.users["admin"] = {
            "salt": salt.hex(),
            "hash": _hash_password(password, salt),
            "role": "sysadmin",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "must_change_password": True,
        }
        self._write()
        print("=" * 64, flush=True)
        print("RIGAL-AI: no users found, created a default Sys Admin account", flush=True)
        print("  username: admin", flush=True)
        print(f"  password: {password}", flush=True)
        print("  (shown once — retrieve again with: docker logs derms-r-orchestrator-1)", flush=True)
        print("  Create a named Sys Admin user from User Management, then delete 'admin'.", flush=True)
        print("=" * 64, flush=True)

    def authenticate(self, username, password):
        user = self.users.get(username)
        if not user or not _verify_password(password, user["salt"], user["hash"]):
            return None
        return {
            "username": username,
            "role": user["role"],
            "must_change_password": bool(user.get("must_change_password")),
        }

    def create_session(self, username, role):
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {"username": username, "role": role}
        return token

    def session_user(self, token):
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            record = self.users.get(session["username"], {})
            return {
                "username": session["username"],
                "role": session["role"],
                "must_change_password": bool(record.get("must_change_password")),
            }

    def destroy_session(self, token):
        with self._lock:
            self._sessions.pop(token, None)

    def list_users(self):
        return [
            {
                "username": name,
                "role": rec["role"],
                "created_at": rec.get("created_at"),
                "must_change_password": bool(rec.get("must_change_password")),
            }
            for name, rec in sorted(self.users.items())
        ]

    def create_user(self, username, password, role):
        username = (username or "").strip()
        if role not in ROLES:
            raise AuthError(f"Unknown role {role!r}. Use 'sysadmin' or 'ml_user'.")
        if not username or not password:
            raise AuthError("Username and password are required.")
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
        with self._lock:
            if username in self.users:
                raise AuthError(f"User {username!r} already exists.")
            salt = secrets.token_bytes(16)
            self.users[username] = {
                "salt": salt.hex(),
                "hash": _hash_password(password, salt),
                "role": role,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "must_change_password": True,
            }
            self._write()

    def change_password(self, username, current_password, new_password):
        if len(new_password or "") < 8:
            raise AuthError("New password must be at least 8 characters.")
        with self._lock:
            user = self.users.get(username)
            if user is None:
                raise AuthError("Unknown user.")
            if not _verify_password(current_password, user["salt"], user["hash"]):
                raise AuthError("Current password is incorrect.")
            salt = secrets.token_bytes(16)
            user["salt"] = salt.hex()
            user["hash"] = _hash_password(new_password, salt)
            user["must_change_password"] = False
            self._write()

    def delete_user(self, username, acting_username):
        if username == acting_username:
            raise AuthError("You cannot delete your own account while logged in as it.")
        with self._lock:
            if username not in self.users:
                raise AuthError(f"User {username!r} not found.")
            remaining_admins = [
                name for name, rec in self.users.items()
                if rec["role"] == "sysadmin" and name != username
            ]
            if self.users[username]["role"] == "sysadmin" and not remaining_admins:
                raise AuthError("Cannot delete the last Sys Admin account.")
            del self.users[username]
            self._write()

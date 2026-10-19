"""SMTP delivery for job notification emails."""
import os
import re
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class MailError(RuntimeError):
    """SMTP rejected the connection, login, or message."""


def parse_emails(raw):
    """Split a free-text field (comma/semicolon/whitespace separated) into validated addresses."""
    if not raw:
        return []
    emails = [part for part in re.split(r"[,;\s]+", raw.strip()) if part]
    invalid = [email for email in emails if not EMAIL_RE.match(email)]
    if invalid:
        raise ValueError(f"Invalid notification email(s): {', '.join(invalid)}")
    # De-duplicate while preserving the order the user typed them in.
    seen = set()
    unique = []
    for email in emails:
        if email not in seen:
            seen.add(email)
            unique.append(email)
    return unique


@dataclass
class Mailer:
    host: str
    port: int
    username: str | None
    password: str | None
    sender: str
    use_tls: bool = True

    @classmethod
    def from_env(cls):
        host = os.environ.get("SMTP_HOST")
        if not host:
            return None
        username = os.environ.get("SMTP_USERNAME") or None
        return cls(
            host=host,
            port=int(os.environ.get("SMTP_PORT", "587")),
            username=username,
            password=os.environ.get("SMTP_PASSWORD") or None,
            sender=os.environ.get("SMTP_FROM") or username or "derms-r-orchestrator@localhost",
            use_tls=os.environ.get("SMTP_USE_TLS", "1") != "0",
        )

    def send(self, to_addrs, subject, body):
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.sender
        message["To"] = ", ".join(to_addrs)
        message.set_content(body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=10) as client:
                if self.use_tls:
                    client.starttls()
                if self.username and self.password:
                    client.login(self.username, self.password)
                client.send_message(message)
        except (OSError, smtplib.SMTPException) as error:
            raise MailError(str(error)) from error

# Security Features

Every request to `/chat` passes through a four-stage validation pipeline before
reaching the RAG agent or the OpenAI API.

---

## Input Validation Pipeline

```
User request
     │
     ▼
1. Sanitize        (max length, null bytes, unicode normalization)
     │
     ▼
2. Prompt injection check   (compiled regex patterns, ~0ms)
     │  → 400 if malicious
     ▼
3. PII detection & redaction (Microsoft Presidio + spaCy, ~50-100ms)
     │  → PII replaced with <TYPE> placeholders
     ▼
4. Rate limiting            (slowapi middleware, per session_id)
     │  → 429 if limit exceeded
     ▼
RAG agent
```

---

## 1. Sanitization

Applied to every request regardless of other feature flags.

| Check | Limit |
|-------|-------|
| Maximum length | 5,000 characters (configurable via `MAX_INPUT_LENGTH`) |
| Null bytes | Stripped |
| Unicode | NFKC-normalized (prevents homoglyph attacks) |
| Whitespace | Stripped from both ends |

Oversized inputs return **HTTP 413 Payload Too Large**.

---

## 2. Prompt Injection Detection

Regex-based detection of known attack patterns. No LLM call — sub-millisecond.

**Blocked patterns:**

| Category | Examples |
|----------|---------|
| Instruction override | "Ignore previous instructions", "Disregard all prior…" |
| System prompt extraction | "Reveal your system message", "What are your instructions?" |
| Role manipulation | "You are now…", "Act as a…", "Pretend you are…" |
| Developer mode | "developer mode", "god mode", "jailbreak" |
| Delimiter injection | `</system>`, `--- end of instructions` |
| Credential extraction | "api_key =", "password:", "token =" |

Detected injection attempts return **HTTP 400 Bad Request** and are logged with
`event: prompt_injection_blocked`.

---

## 3. PII Detection & Redaction

Uses [Microsoft Presidio](https://microsoft.github.io/presidio/) with spaCy's
`en_core_web_sm` model.

**Entities detected and redacted:**

- `CREDIT_CARD` — Credit card numbers
- `US_SSN` — Social Security Numbers
- `PHONE_NUMBER` — Phone numbers
- `EMAIL_ADDRESS` — Email addresses
- `IP_ADDRESS` — IPv4 / IPv6 addresses
- `PERSON` — Personal names
- `LOCATION` — Street addresses and locations

Redacted text looks like: `"My SSN is <US_SSN> and email is <EMAIL_ADDRESS>"`

The audit log records the entity **type** and **position** — never the raw value.

---

## 4. Rate Limiting

Rate limit key: `session_id` from the request body (falls back to client IP).

| Limit | Value |
|-------|-------|
| Per minute | 10 requests |
| Per hour | 100 requests |

Exceeded limits return **HTTP 429 Too Many Requests**.

---

## Configuration

All security features are controlled by environment variables:

```bash
# .env
SECURITY_ENABLE_PROMPT_INJECTION_CHECK=true
SECURITY_ENABLE_PII_DETECTION=true
SECURITY_ENABLE_RATE_LIMITING=true
MAX_INPUT_LENGTH=5000
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_PER_HOUR=100
```

**Disabling in development only:**

```bash
# .env (local dev / load testing — NEVER in production)
SECURITY_ENABLE_PROMPT_INJECTION_CHECK=false
SECURITY_ENABLE_PII_DETECTION=false
```

---

## Structured Log Events

| Event | Level | Triggered by |
|-------|-------|--------------|
| `input_validation_passed` | info | Every clean request |
| `prompt_injection_detected` | warning | Regex match |
| `prompt_injection_blocked` | error | Injection attempt blocked |
| `pii_detected_and_redacted` | warning | PII found in input |
| `pii_redacted_from_input` | warning | PII removed before OpenAI call |

---

## Security Incident Response

If you see repeated injection attempts:

1. Search logs: `grep "prompt_injection_blocked" logs/`
2. Identify the `session_id`
3. Review patterns for false positives
4. Adjust regex in `src/security/prompt_injection.py` if needed

---

## Reporting Vulnerabilities

Do not open a public GitHub issue for security vulnerabilities.
Contact the maintainer directly with a description and reproduction steps.

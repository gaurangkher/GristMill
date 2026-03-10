# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest `main` | ✅ |
| Previous minor | ✅ security fixes only |
| Older | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately via [GitHub Security Advisories](https://github.com/OWNER/GristMill/security/advisories/new).

Include:
- Description of the vulnerability and its impact
- Steps to reproduce
- Affected component and version
- Suggested fix if you have one

We will acknowledge your report within **48 hours** and aim to release a fix within **14 days** for critical issues.

## Scope

### In scope

- Rust core crates (memory corruption, unsafe code misuse, IPC injection)
- LLM prompt injection via untrusted event payloads
- Token budget bypass in `grist-hammer`
- Path traversal in model file loading (`grist-grinders`)
- Secret leakage via logs (`ANTHROPIC_API_KEY`, `SLACK_WEBHOOK_URL`, etc.)
- Plugin sandbox escapes (TypeScript plugin system)

### Out of scope

- Vulnerabilities in third-party dependencies (report to upstream)
- DoS via resource exhaustion under authenticated access
- Issues requiring local machine access

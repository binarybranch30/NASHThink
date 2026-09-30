# Security policy

NASH Think holds people's private chats, so we take reports seriously.

## Reporting a vulnerability

Please **do not open a public GitHub issue** for security problems.

- Use GitHub's private reporting: **Security → Report a vulnerability** on
  [github.com/binarybranch30/nashthink](https://github.com/binarybranch30/nashthink/security/advisories/new), or
- email **naitiksri08@gmail.com** with the subject `NASH Think security`.

Include what you found, how to reproduce it, and what it exposes. Never include real chat data; the fictional
archive in `demo/` is enough to show any issue. We aim to reply within 72 hours and to fix confirmed issues
before disclosing them. We'll credit you unless you'd rather we didn't.

## Scope and trust model

- The server is meant to listen on `127.0.0.1` only. Exposing it on a network is out of scope unless the report
  shows a weakness in workspace passwords or sessions.
- In scope: bypassing the workspace password, reading another workspace's data, path traversal in uploads or
  file serving, leaking `.env` or tokens.
- Out of scope: attacks that need code execution as the same OS user (such a process can read the data files directly).

## Supported versions

Only the latest `main` gets fixes.

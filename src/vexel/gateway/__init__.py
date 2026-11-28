"""Vexel AI gateway: the only way the desktop app reaches hosted language models (0.8 plan,
phase 3).

The gateway runs on a server Vexel operates (default: Singapore, reachable from mainland China
and abroad). It holds the providers' API keys - the app never sees one - and answers exactly one
kind of request: *plan the measurement of this image*. There is no chat.

- ``store``: organisations, invite codes, users and their tokens (hashed), the usage ledger
  (counts and tokens, never images or goal text) and the response cache, in SQLite.
- ``guard``: what may reach a model - the goal (length, off-topic, prompt injection, requests
  for secrets, abuse) and the structured context (allowed fields, sizes, numbers).
- ``providers``: DeepSeek and Gemini over HTTPS with a hard output-token cap and the real token
  counts from each reply; a simulated provider for development and tests.
- ``service``: one planning request end to end - quota, guard, cache, prompt built on the
  server from Vexel's own recipe definitions, routing by region with fallback, answer checked
  against the schema and the allowlist, usage recorded.
- ``app``: the HTTP API (``vexel gateway serve``); ``admin``: organisations, invites, usage.
"""

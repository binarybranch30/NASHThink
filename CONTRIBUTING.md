# Contributing to NASH Think

Thanks for helping. NASH Think is a local-first, privacy-first project: keep changes local by default, and never
commit real chat data.

## Workflow

1. Fork, then branch from `main` (`feature/<short-name>`).
2. Make a focused change with tests (`scripts/tests/test_<area>.py`; use fake models and throwaway data, as the
   existing suites do).
3. Run the checks below; CI runs the same ones on every push.
4. Open a pull request that says what changed, why, and how you tested it. Screenshots must use the fictional sample archive.

## Checks

```bash
.venv/bin/ruff check scripts --select E9,F63,F7,F82
for t in scripts/tests/test_*.py; do .venv/bin/python "$t" || echo "FAILED: $t"; done
```

## Code style

- Python ≥ 3.11, standard library first; add a dependency only when it earns its place (and add it to `requirements.txt`).
- Match the surrounding code: small functions, a docstring that says *why*, plain names, 120-column lines.
- The memory database is read-only for the server; new state goes in its own store under `processed_data/`, created owner-only (`0600`) if it can hold chat text.
- Nothing leaves the machine by default. Online features must be opt-in and labelled in the UI.
- Anything an AI agent can change must go through the action queue (`scripts/api/actions.py`), never straight to a service.
- Tests and logs never print archive text: counts, ids and booleans only.

## Commit messages

`Area: what changed, in plain words` (e.g. `Reminders: snooze until tomorrow morning`), with details in the body when needed.

## License

By contributing you agree that your contribution is released under the [MIT License](LICENSE).

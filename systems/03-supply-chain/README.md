# Exercise 3 (Solution): When a Source Goes Dark

The complete project: the timeout path and the resilient coordinator. This stage
equals the reference solution.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -q
.venv/bin/supply-chain-investigate meridian --offline
```

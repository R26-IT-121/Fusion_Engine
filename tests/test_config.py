import os
import sys

sys.path.insert(0, r"C:\Projects\DeepSentinel")

from backend import config

print("--- resolved from config.ini ---")
print("graph_api_base :", config.get("upstream", "graph_api_base"))
tmo = config.get("upstream", "timeout_ms")
print("timeout_ms     :", tmo, "->", type(tmo).__name__)
print("session mins   :", config.get("auth", "access_token_expire_minutes"))
print("users_db path  :", config.get_path("paths", "users_db"))
print("gemini key set :", bool(config.get("secrets", "gemini_api_key")))
print("problems       :", config.validate(strict=False))

# environment must take precedence over config.ini
os.environ["GRAPH_API_BASE"] = "http://from-env:9999"
config.reload()
resolved = config.get("upstream", "graph_api_base")
print("\n--- after setting GRAPH_API_BASE env var ---")
print("graph_api_base :", resolved)
assert resolved == "http://from-env:9999", "env override FAILED"
print("env override OK")

# secrets must be masked in describe()
os.environ.pop("GRAPH_API_BASE")
config.reload()
desc = config.describe()
assert "AIzaSy" not in desc, "SECRET LEAKED into describe()"
assert "mfdUDDU9" not in desc, "SECRET LEAKED into describe()"
print("secrets masked in describe() OK")

# bad cast must raise a clear error
os.environ["UPSTREAM_TIMEOUT_MS"] = "not-a-number"
config.reload()
try:
    config.get("upstream", "timeout_ms")
    print("FAIL: bad int accepted")
except ValueError as e:
    print("bad value rejected OK ->", e)
os.environ.pop("UPSTREAM_TIMEOUT_MS")

"""End-to-end test of DB-backed auth and role enforcement."""

import sys

sys.path.insert(0, r"C:\Projects\DeepSentinel")

from fastapi.testclient import TestClient

from backend.main import app

FAILURES = []


def check(name, cond, extra=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {extra}")
        FAILURES.append(name)


with TestClient(app) as c:
    print("\n=== 1. Unauthenticated access is refused ===")
    for method, path in [
        ("get", "/settings"),
        ("get", "/users"),
        ("get", "/audit-log"),
        ("post", "/settings/risk-manager"),
    ]:
        r = c.post(path, json={}) if method == "post" else c.get(path)
        check(f"{method.upper()} {path} -> 401", r.status_code == 401, f"got {r.status_code}")

    print("\n=== 2. Bad credentials ===")
    r = c.post("/auth/login", json={"username": "admin", "password": "wrong"})
    check("wrong password -> 401", r.status_code == 401, f"got {r.status_code}")
    r = c.post("/auth/login", json={"username": "nobody", "password": "x"})
    check("unknown user -> 401", r.status_code == 401, f"got {r.status_code}")
    check(
        "identical message (no user enumeration)",
        r.json().get("detail") == "Invalid username or password",
        r.json(),
    )

    print("\n=== 3. Admin login ===")
    r = c.post("/auth/login", json={"username": "admin", "password": "admin123"})
    check("login -> 200", r.status_code == 200, r.text[:200])
    admin_token = r.json()["access_token"]
    admin_h = {"Authorization": f"Bearer {admin_token}"}
    check("role is admin", r.json()["user"]["role"] == "admin")

    print("\n=== 4. Admin can reach admin-only endpoints ===")
    r = c.get("/users", headers=admin_h)
    check("GET /users -> 200", r.status_code == 200, r.text[:200])
    r = c.get("/audit-log", headers=admin_h)
    check("GET /audit-log -> 200", r.status_code == 200)
    check("audit log recorded the logins", len(r.json()) > 0)

    print("\n=== 5. Admin creates a risk manager and an analyst ===")
    r = c.post("/users", headers=admin_h, json={
        "username": "rmanager", "email": "rm@bank.com", "full_name": "Bank Risk Manager",
        "password": "rmpassword1", "role": "risk_manager"})
    check("create risk_manager -> 201", r.status_code == 201, r.text[:200])

    r = c.post("/users", headers=admin_h, json={
        "username": "analyst1", "email": "an@bank.com", "full_name": "Assistant Manager",
        "password": "anpassword1", "role": "analyst"})
    check("create analyst -> 201", r.status_code == 201, r.text[:200])

    r = c.post("/users", headers=admin_h, json={
        "username": "rmanager", "email": "other@bank.com", "full_name": "Dup",
        "password": "password123", "role": "analyst"})
    check("duplicate username -> 409", r.status_code == 409, f"got {r.status_code}")

    r = c.post("/users", headers=admin_h, json={
        "username": "shortpw", "email": "s@bank.com", "full_name": "Short",
        "password": "abc", "role": "analyst"})
    check("short password -> 422", r.status_code == 422, f"got {r.status_code}")

    print("\n=== 6. Risk manager permissions ===")
    r = c.post("/auth/login", json={"username": "rmanager", "password": "rmpassword1"})
    check("risk manager login -> 200", r.status_code == 200, r.text[:200])
    rm_h = {"Authorization": f"Bearer {r.json()['access_token']}"}

    r = c.get("/settings", headers=rm_h)
    check("CAN read settings -> 200", r.status_code == 200, f"got {r.status_code}")

    r = c.post("/settings/risk-manager", headers=rm_h,
               json={"name": "Alerts", "email": "alerts@bank.com", "role": "Risk Manager"})
    check("CAN add alert recipient -> 201", r.status_code == 201, r.text[:200])

    r = c.get("/users", headers=rm_h)
    check("CANNOT list users -> 403", r.status_code == 403, f"got {r.status_code}")

    r = c.get("/audit-log", headers=rm_h)
    check("CANNOT read audit log -> 403", r.status_code == 403, f"got {r.status_code}")

    print("\n=== 7. Analyst permissions (read-only) ===")
    r = c.post("/auth/login", json={"username": "analyst1", "password": "anpassword1"})
    check("analyst login -> 200", r.status_code == 200, r.text[:200])
    an_h = {"Authorization": f"Bearer {r.json()['access_token']}"}

    r = c.get("/auth/me", headers=an_h)
    check("CAN read own profile -> 200", r.status_code == 200)

    r = c.get("/settings", headers=an_h)
    check("CANNOT read settings -> 403", r.status_code == 403, f"got {r.status_code}")

    r = c.post("/settings/risk-manager", headers=an_h,
               json={"name": "X", "email": "x@bank.com", "role": "Risk Manager"})
    check("CANNOT add alert recipient -> 403", r.status_code == 403, f"got {r.status_code}")

    r = c.get("/users", headers=an_h)
    check("CANNOT list users -> 403", r.status_code == 403, f"got {r.status_code}")

    print("\n=== 8. Tampered and malformed tokens ===")
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {admin_token[:-4]}XXXX"})
    check("tampered signature -> 401", r.status_code == 401, f"got {r.status_code}")
    r = c.get("/auth/me", headers={"Authorization": "Bearer garbage"})
    check("garbage token -> 401", r.status_code == 401, f"got {r.status_code}")
    r = c.get("/auth/me", headers={"Authorization": admin_token})
    check("missing Bearer scheme -> 401", r.status_code == 401, f"got {r.status_code}")

    print("\n=== 9. Password change invalidates old sessions ===")
    r = c.post("/auth/login", json={"username": "analyst1", "password": "anpassword1"})
    old_h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = c.post("/auth/change-password", headers=old_h,
               json={"current_password": "anpassword1", "new_password": "newpassword9"})
    check("change password -> 200", r.status_code == 200, r.text[:200])

    r = c.post("/auth/login", json={"username": "analyst1", "password": "newpassword9"})
    check("login with new password -> 200", r.status_code == 200)
    r = c.post("/auth/login", json={"username": "analyst1", "password": "anpassword1"})
    check("old password rejected -> 401", r.status_code == 401, f"got {r.status_code}")

    print("\n=== 10. Self-protection guards ===")
    r = c.delete("/users/admin", headers=admin_h)
    check("admin cannot delete self -> 409", r.status_code == 409, f"got {r.status_code}")
    r = c.patch("/users/admin/enabled", headers=admin_h, json={"enabled": False})
    check("admin cannot disable self -> 409", r.status_code == 409, f"got {r.status_code}")

    print("\n=== 11. Account lockout after repeated failures ===")
    for _ in range(5):
        c.post("/auth/login", json={"username": "rmanager", "password": "wrong"})
    r = c.post("/auth/login", json={"username": "rmanager", "password": "rmpassword1"})
    check("locked after 5 failures -> 423", r.status_code == 423, f"got {r.status_code}")

    print("\n=== 12. Input validation ===")
    r = c.post("/settings/alert-settings", headers=admin_h, json={"fraud_threshold": 5.0})
    check("threshold out of range -> 422", r.status_code == 422, f"got {r.status_code}")
    r = c.post("/settings/alert-settings", headers=admin_h, json={"nonsense_key": 1})
    check("unknown setting key -> 422", r.status_code == 422, f"got {r.status_code}")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")

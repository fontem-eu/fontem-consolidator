def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_rules_endpoint_returns_list(client):
    c, _ = client
    r = c.get("/rules")
    assert r.status_code == 200
    rules = r.json()
    assert isinstance(rules, list)
    assert len(rules) >= 8  # Company(5) + Authority(3)
    names = {r["name"] for r in rules}
    assert "exact_lei_match" in names
    assert "exact_authority_id_match" in names

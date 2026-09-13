import pytest

from harness.registry import ToolRegistry, ToolSpec, validate_args
from harness.todo import apply_update, open_ids, validate_todos


def _spec(name):
    return ToolSpec(name, f"{name} does things\nmore detail", {"type": "object", "properties": {}, "required": []}, lambda: name)


def test_register_and_list():
    r = ToolRegistry()
    for i in range(3):
        r.register(_spec(f"t{i}"))
    assert r.names() == ["t0", "t1", "t2"]
    assert r.list() == [{"name": f"t{i}", "description": f"t{i} does things"} for i in range(3)]
    assert r.list("t1") == [{"name": "t1", "description": "t1 does things"}]
    with pytest.raises(ValueError):
        r.register(_spec("t0"))


def test_decorator_registers_and_returns_fn():
    r = ToolRegistry()

    @r.tool("add", "Add", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]})
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert r.get("add").fn(a=2, b=3) == 5
    assert r.get("add").schema()["function"]["name"] == "add"


def test_validate_args():
    params = {"type": "object", "properties": {"a": {"type": "integer"}, "s": {"type": "string"}},
              "required": ["a"], "additionalProperties": False}
    assert validate_args(params, {"a": 1}) is None
    assert "missing required" in validate_args(params, {})
    assert "unknown argument" in validate_args(params, {"a": 1, "z": 2})
    assert "must be integer" in validate_args(params, {"a": "1"})
    assert "must be integer" in validate_args(params, {"a": True})
    assert "must be a JSON object" in validate_args(params, "not json")


def test_todo_validate_and_merge():
    assert validate_todos("x") == "todos must be a list"
    assert "status" in validate_todos([{"id": "1", "content": "x", "status": "bogus"}])
    assert "duplicate" in validate_todos([{"id": "1", "content": "x", "status": "pending"}] * 2)
    cur = [{"id": "1", "content": "a", "status": "pending"}, {"id": "2", "content": "b", "status": "pending"}]
    merged = apply_update(cur, [{"id": "2", "content": "b", "status": "completed"},
                                {"id": "3", "content": "c", "status": "pending"}], merge=True)
    assert [t["id"] for t in merged] == ["1", "2", "3"]
    assert open_ids(merged) == ["1", "3"]
    replaced = apply_update(cur, [{"id": "9", "content": "z", "status": "cancelled"}], merge=False)
    assert [t["id"] for t in replaced] == ["9"]
    assert open_ids(replaced) == []


def test_filter_matches_exactly_what_the_listing_shows():
    r = ToolRegistry()
    r.register(ToolSpec("alpha", "Do alpha things.\nDetail line nobody sees mentions zebras.",
                        {"type": "object", "properties": {}, "required": []}, lambda: "a"))
    r.register(ToolSpec("zebra_count", "Count stripes.",
                        {"type": "object", "properties": {}, "required": []}, lambda: "z"))
    # the needle is matched against the name and the first line, which is all
    # `list` returns; a hit on a hidden line would be a hit the model cannot see
    assert [e["name"] for e in r.list("zebra")] == ["zebra_count"]
    assert r.list("alpha things") == [{"name": "alpha", "description": "Do alpha things."}]
    assert r.list("ALPHA") == [{"name": "alpha", "description": "Do alpha things."}]
    assert r.list("nothing here") == []
    assert r.get("alpha").summary() == "Do alpha things."

"""Tests for the resilient extraction schema."""
from __future__ import annotations

from mortgage_extractor import schema, tools


def test_ac_01_01_tool_definition_shape() -> None:
    tool = tools.extract_mortgage_data()
    assert tool["name"] == "extract_mortgage_data"
    assert isinstance(tool["description"], str) and tool["description"].strip()
    input_schema = tool["input_schema"]
    assert input_schema["type"] == "object"
    assert isinstance(input_schema["properties"], dict)
    assert len(input_schema["properties"]) > 0


def test_ac_01_02_required_and_optional_fields() -> None:
    s = schema.mortgage_data_schema()

    assert "full_name" in s["properties"]["borrower"]["required"]
    assert "address" in s["properties"]["property"]["required"]
    assert "amount" in s["properties"]["loan"]["required"]

    assert "coborrower_name" in s["properties"]["borrower"]["properties"]
    assert "coborrower_name" not in s["properties"]["borrower"].get("required", [])
    assert "year_built" in s["properties"]["property"]["properties"]
    assert "year_built" not in s["properties"]["property"].get("required", [])


def test_ac_01_03_enum_other_with_detail_field() -> None:
    s = schema.mortgage_data_schema()
    prop_type = s["properties"]["property"]["properties"]["property_type"]
    assert "enum" in prop_type
    assert "other" in prop_type["enum"]

    detail_field = s["properties"]["property"]["properties"]["property_type_detail"]
    assert detail_field is not None
    detail_type = detail_field["type"]
    assert detail_type == "string" or "string" in detail_type


def test_ac_01_04_at_least_three_nullable_fields() -> None:
    s = schema.mortgage_data_schema()
    nullable_paths = schema.list_nullable_fields(s)
    assert len(nullable_paths) >= 3

    for path in [
        "borrower.coborrower_name",
        "property.hoa_dues_monthly",
        "income.bonus_ytd",
    ]:
        assert path in nullable_paths, f"expected nullable: {path}"


def test_ac_01_05_list_nullable_fields_returns_dotted_paths() -> None:
    s = schema.mortgage_data_schema()
    paths = schema.list_nullable_fields(s)
    assert all(isinstance(p, str) for p in paths)
    assert all("." in p for p in paths), "expected dotted paths"
    for p in paths:
        cursor: object = s
        for segment in p.split("."):
            assert isinstance(cursor, dict)
            cursor = cursor["properties"][segment]
        assert isinstance(cursor, dict)
        type_field = cursor["type"]
        assert "null" in type_field

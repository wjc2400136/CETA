def canonical_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")

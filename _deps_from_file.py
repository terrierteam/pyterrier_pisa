"""scikit-build-core custom metadata provider: reads dependencies from requirements.txt."""


def dynamic_metadata(field, settings=None):
    if field != "dependencies":
        raise ValueError(f"Unknown field: {field}")
    with open("requirements.txt") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

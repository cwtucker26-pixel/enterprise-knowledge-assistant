"""Join doc_metadata.csv onto documents and normalise system names."""
import csv
import sys

from src.config import RAW_DIR

# Canonical spellings; doc_metadata.csv has Auth-Db, Apigateway, Userprofile-Api, Datawarehouse.
CANONICAL_SYSTEMS = [
    "Payment-Service",
    "Auth-DB",
    "UserProfile-API",
    "DataWarehouse",
    "APIGateway",
    "Notification-Service",
    "ReportingPortal",
    "InventoryEngine",
]
_BY_LOWER = {name.lower(): name for name in CANONICAL_SYSTEMS}


def normalise_system(name: str) -> str:
    name = name.strip()
    canonical = _BY_LOWER.get(name.lower())
    if canonical is None:
        print(f"WARNING: unknown system name '{name}' left unchanged", file=sys.stderr)
        return name
    return canonical


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def load_metadata_table() -> dict[str, dict]:
    """filename -> metadata dict (system_refs / sop_refs as lists, systems normalised)."""
    table = {}
    with open(RAW_DIR / "doc_metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["filename"]] = {
                "source_file": row["filename"],
                "doc_type": row["doc_type"],
                "dept": row["dept"],
                "date": row["date"],
                "author": row["author"],
                "system_refs": [normalise_system(s) for s in _split(row["system_refs"])],
                "sop_refs": _split(row["sop_refs"]),
            }
    return table

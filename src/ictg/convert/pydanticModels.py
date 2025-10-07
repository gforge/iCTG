from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HrBlock(BaseModel):
    Values: List[int] = Field(default_factory=list)
    SignalQuality: Optional[str] = None  # only present for Hr1/Hr2


class TocoBlock(BaseModel):
    # appears to be base64; keep as str (decode downstream if needed)
    Values: str


class PatientRecord(BaseModel):
    PatientID: str
    RegistrationID: int
    Timestamp: datetime
    Hr1Mode: Optional[str] = None
    Hr2Mode: Optional[str] = None
    MhrMode: Optional[str] = None
    TocoMode: Optional[str] = None
    Hr1: Optional[HrBlock] = None
    Hr2: Optional[HrBlock] = None
    Mhr: Optional[HrBlock] = None
    Toco: Optional[TocoBlock] = None

    @field_validator("Timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        """Parse timestamp from the format: 7/24/2021 1:16:02 PM"""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.strptime(v, "%m/%d/%Y %I:%M:%S %p")
        raise ValueError(f"Invalid timestamp format: {v}")


# ---------- normalization / flattening ----------
def normalize_patient_record(rec: PatientRecord) -> Dict[str, Any]:
    """
    Convert a validated PatientRecord into a flat dict suitable for DataFrame rows.
    - Timestamp is already a datetime object from Pydantic validation
    - Flatten Hr1/Hr2/Mhr Values into fixed columns 0..3 if present, else keep empty.
    - Keep Toco.Values as base64 string (optionally decode elsewhere).
    """

    def values_cols(block: Optional[HrBlock], prefix: str) -> Dict[str, Any]:
        vals = block.Values if block else []
        # Expand first 4 positions; adjust if your data may vary in length
        cols: dict[str, int | str | None] = {
            f"{prefix}_{i}": (vals[i] if i < len(vals) else None) for i in range(4)
        }
        if block and block.SignalQuality is not None:
            cols[f"{prefix}_SignalQuality"] = block.SignalQuality
        else:
            cols[f"{prefix}_SignalQuality"] = None
        return cols

    row: Dict[str, Any] = {
        "PatientID": rec.PatientID,
        "RegistrationID": rec.RegistrationID,
        "Timestamp": rec.Timestamp,  # already a datetime object
        "Hr1Mode": rec.Hr1Mode,
        "Hr2Mode": rec.Hr2Mode,
        "MhrMode": rec.MhrMode,
        "TocoMode": rec.TocoMode,
        "Toco_Values": rec.Toco.Values if rec.Toco else None,
    }
    row.update(values_cols(rec.Hr1, "Hr1"))
    row.update(values_cols(rec.Hr2, "Hr2"))
    row.update(values_cols(rec.Mhr, "Mhr"))
    return row

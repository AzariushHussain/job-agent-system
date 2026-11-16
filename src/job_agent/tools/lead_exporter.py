import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

LEAD_COLUMNS = [
    "idempotency_key",
    "timestamp_utc",
    "company_name",
    "role_title",
    "location",
    "tech_match_score",
    "matched_skills",
    "matched_experiences",
    "matched_projects",
    "recipient_role",
    "target_email",
    "email_subject",
    "email_body",
    "linkedin_note",
    "static_resume_path",
    "job_url",
]


def compute_lead_idempotency_key(company_name: str, role_title: str) -> str:
    normalized = f"{company_name.strip().lower()}::{role_title.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def export_vetted_lead(
    lead_row: Dict[str, Any],
    output_dir: str | Path = "outputs",
) -> Tuple[str, Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vetted_job_leads.csv"
    xlsx_path = out_dir / "vetted_job_leads.xlsx"

    idem_key = lead_row.get("idempotency_key") or compute_lead_idempotency_key(
        str(lead_row.get("company_name", "")),
        str(lead_row.get("role_title", "")),
    )
    lead_row["idempotency_key"] = idem_key
    lead_row.setdefault(
        "timestamp_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )

    existing_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("idempotency_key"):
                    existing_keys.add(row["idempotency_key"])

    if idem_key in existing_keys:
        return "DUPLICATE_SKIPPED", csv_path, xlsx_path

    ordered_values = [str(lead_row.get(col, "")) for col in LEAD_COLUMNS]

    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LEAD_COLUMNS)
        writer.writerow(ordered_values)

    if xlsx_path.exists():
        wb = load_workbook(xlsx_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Vetted Outreach Leads"
        ws.append(LEAD_COLUMNS)
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        for col_idx in range(1, len(LEAD_COLUMNS) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.append(ordered_values)
    wb.save(xlsx_path)

    return "EXPORTED", csv_path, xlsx_path

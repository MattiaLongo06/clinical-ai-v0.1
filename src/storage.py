# src/storage.py
"""
Utility di persistenza per Clinical Imaging v0.1.
- Salvataggio riga CSV con metadati paziente
- Prevenzione duplicati ravvicinati
"""

from __future__ import annotations
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple, List, Dict


CSV_HEADER = ["timestamp", "filename", "age", "sex", "symptoms"]


def _ensure_csv(path: Path) -> None:
    """Crea cartella e CSV con header se non esistono."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def _read_tail_rows(path: Path, max_rows: int = 200) -> List[Dict[str, str]]:
    """Ritorna le ultime `max_rows` righe del CSV come dict (senza header)."""
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # leggere tutto è ok su file piccoli; per sicurezza tagliamo la coda
        rows = list(reader)[-max_rows:]
    return rows


def _symptoms_to_str(symptoms: Iterable[str]) -> str:
    """Normalizza i sintomi in una stringa stabile."""
    if symptoms is None:
        return ""
    # ordiniamo così 'Fever;Cough' = 'Cough;Fever' -> meno duplicati
    return ";".join(sorted([s.strip() for s in symptoms if s is not None and str(s).strip() != ""]))


def save_patient_row(
    age: int,
    sex: str,
    symptoms: Iterable[str],
    filename: str,
    data_dir: str = "data",
    min_seconds_between_same_file: int = 20,
) -> Tuple[str, bool]:
    """
    Aggiunge (timestamp, filename, age, sex, symptoms) a data/patient_data.csv.

    Evita di scrivere duplicati ravvicinati (stessi campi) se l’ultima riga
    identica è stata scritta meno di `min_seconds_between_same_file` secondi fa.
    Ritorna (csv_path, saved) dove `saved` indica se abbiamo realmente scritto.

    Esempio:
        path, saved = save_patient_row(46, "Female", ["Fever", "Chest pain"], "chest.dcm")
    """
    csv_path = Path(data_dir) / "patient_data.csv"
    _ensure_csv(csv_path)

    now = datetime.now()
    ts_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    symptoms_str = _symptoms_to_str(symptoms)

    # Controllo duplicati su coda recente
    recent = _read_tail_rows(csv_path, max_rows=200)
    for row in reversed(recent):
        if (
            row.get("filename") == filename
            and row.get("age") == str(age)
            and (row.get("sex") or "") == (sex or "")
            and (row.get("symptoms") or "") == symptoms_str
        ):
            # stessa riga: controlliamo quanto tempo è passato
            try:
                prev_ts = datetime.fromisoformat(row.get("timestamp", ""))
                delta = (now - prev_ts).total_seconds()
                if 0 <= delta < float(min_seconds_between_same_file):
                    # troppo ravvicinato: non salviamo
                    return (str(csv_path), False)
            except Exception:
                # se timestamp malformato, ignoriamo il controllo tempo
                pass
            # se la riga identica non è “ravvicinata”, continuiamo e salviamo

            break  # abbiamo già verificato l’ultimo match: usciamo dal loop

    # Scrittura
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts_str, filename, age, sex or "", symptoms_str])

    return (str(csv_path), True)


# ---- helper opzionali ------------------------------------------------------

def read_patient_csv(data_dir: str = "data") -> List[Dict[str, str]]:
    """Legge tutto il CSV come lista di dict (header compreso)."""
    path = Path(data_dir) / "patient_data.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reset_patient_csv(data_dir: str = "data") -> str:
    """Svuota il CSV mantenendo l'header. Ritorna il percorso del file."""
    path = Path(data_dir) / "patient_data.csv"
    _ensure_csv(path)
    # riscrive header e basta
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    return str(path)

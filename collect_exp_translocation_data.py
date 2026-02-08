#!/usr/bin/env python3
"""
Collect experimental translocation data and add to database.
Matches worm_id to worm lengths from CSV files in EXP_DATA_Cavity.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent.parent / "CONFINEMENT" / "N_40"
DATA_EXP_DIR = BASE_DIR / "DATA_EXP"
EXP_DATA_CAVITY_DIR = BASE_DIR / "EXP_DATA_Cavity"
DB_PATH = Path(__file__).parent / "exp_trapping_times.db"

TEMPERATURES = [10, 20, 30]


def load_worm_lengths():
    """Load worm lengths from CSV files in EXP_DATA_Cavity."""
    worm_lengths = {}  # {(exp_folder, worm_id): (length_ref, temp)}

    if not EXP_DATA_CAVITY_DIR.exists():
        print(f"Warning: {EXP_DATA_CAVITY_DIR} not found")
        return worm_lengths

    exp_folders = [f for f in EXP_DATA_CAVITY_DIR.iterdir() if f.is_dir() and not f.name.startswith(".")]

    for exp_folder in exp_folders:
        csv_files = list(exp_folder.glob("worm_*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0)
                if len(df) == 0:
                    continue

                temp = df["T"].iloc[0]
                length_ref = df["length_ref"].iloc[0]
                worm_id = csv_file.stem

                key = (exp_folder.name, worm_id)
                worm_lengths[key] = (length_ref, temp)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    return worm_lengths


def load_translocation_events_and_metrics(temp, worm_lengths):
    """Load translocation events and compute per-worm metrics."""
    trans_file = DATA_EXP_DIR / f"translocation_times_T{temp}.txt"

    if not trans_file.exists():
        print(f"File not found: {trans_file}")
        return [], {}

    events = []
    worm_stats = {}  # {(exp_folder, worm_id): {"n_success": 0, "n_attempt": 0}}
    total_obs_time = None

    # First pass: collect events and extract observation time
    with open(trans_file, "r") as f:
        for line in f:
            # Extract observation time from comments
            if "Total observation time:" in line:
                total_obs_time = float(line.split(":")[1].split("min")[0].strip())
                continue

            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split(",")
            if len(parts) < 4:
                continue

            time_val = float(parts[0])
            worm_path = parts[1].strip()
            event_type = parts[3].strip()

            # Parse worm path
            path_parts = worm_path.split("/")
            if len(path_parts) != 2:
                continue

            exp_folder, worm_id = path_parts
            key = (exp_folder, worm_id)

            # Get worm length
            if key not in worm_lengths:
                continue

            length_ref, _ = worm_lengths[key]

            events.append({
                "T_exp": temp,
                "length_mm": length_ref,
                "time_min": time_val,
                "event_type": event_type,
                "worm_key": f"{exp_folder}/{worm_id}",
            })

            # Track per-worm stats
            if key not in worm_stats:
                worm_stats[key] = {"n_success": 0, "n_attempt": 0, "length_mm": length_ref}
            if event_type == "success":
                worm_stats[key]["n_success"] += 1
            elif event_type == "attempt":
                worm_stats[key]["n_attempt"] += 1

    # Compute per-worm metrics
    worm_metrics = {}
    n_worms = len(worm_stats)
    obs_time_per_worm = total_obs_time / n_worms if total_obs_time and n_worms > 0 else 0

    for key, stats in worm_stats.items():
        worm_metrics[key] = {
            "T_exp": temp,
            "worm_key": f"{key[0]}/{key[1]}",
            "length_mm": stats["length_mm"],
            "obs_time_min": obs_time_per_worm,
            "n_success": stats["n_success"],
            "n_attempt": stats["n_attempt"],
        }

    return events, worm_metrics


def main():
    print("=" * 60)
    print("Collecting experimental translocation data")
    print("=" * 60)

    # Load worm lengths
    print("\nLoading worm lengths...")
    worm_lengths = load_worm_lengths()
    print(f"Loaded {len(worm_lengths)} worms")

    # Collect all events and worm metrics
    all_events = []
    all_worm_metrics = {}

    for temp in TEMPERATURES:
        print(f"\nProcessing T={temp}°C...")
        events, worm_metrics = load_translocation_events_and_metrics(temp, worm_lengths)
        print(f"  Found {len(events)} events, {len(worm_metrics)} worms")
        all_events.extend(events)
        all_worm_metrics.update(worm_metrics)

    print(f"\nTotal events: {len(all_events)}")

    # Count by type
    success_count = sum(1 for e in all_events if e["event_type"] == "success")
    attempt_count = sum(1 for e in all_events if e["event_type"] == "attempt")
    print(f"  Success: {success_count}")
    print(f"  Attempt: {attempt_count}")
    print(f"\nTotal worms with metrics: {len(all_worm_metrics)}")

    # Add to database
    print(f"\nAdding to database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create translocation events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exp_translocation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            T_exp INTEGER NOT NULL,
            length_mm REAL NOT NULL,
            time_min REAL NOT NULL,
            event_type TEXT NOT NULL
        )
    """)

    # Create worm metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exp_worm_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            T_exp INTEGER NOT NULL,
            worm_key TEXT NOT NULL,
            length_mm REAL NOT NULL,
            obs_time_min REAL NOT NULL,
            n_success INTEGER NOT NULL,
            n_attempt INTEGER NOT NULL
        )
    """)

    # Clear existing data
    cursor.execute("DELETE FROM exp_translocation_events")
    cursor.execute("DELETE FROM exp_worm_metrics")
    print("  Cleared existing data")

    # Insert translocation events
    cursor.executemany(
        "INSERT INTO exp_translocation_events (T_exp, length_mm, time_min, event_type) VALUES (?, ?, ?, ?)",
        [(e["T_exp"], e["length_mm"], e["time_min"], e["event_type"]) for e in all_events],
    )

    # Insert worm metrics
    cursor.executemany(
        "INSERT INTO exp_worm_metrics (T_exp, worm_key, length_mm, obs_time_min, n_success, n_attempt) VALUES (?, ?, ?, ?, ?, ?)",
        [(m["T_exp"], m["worm_key"], m["length_mm"], m["obs_time_min"], m["n_success"], m["n_attempt"]) for m in all_worm_metrics.values()],
    )

    conn.commit()

    # Verify events
    cursor.execute("SELECT COUNT(*) FROM exp_translocation_events")
    count = cursor.fetchone()[0]
    print(f"  Inserted {count} events")

    # Verify worm metrics
    cursor.execute("SELECT COUNT(*) FROM exp_worm_metrics")
    count = cursor.fetchone()[0]
    print(f"  Inserted {count} worm metrics")

    # Summary by temperature
    print("\nSummary by temperature:")
    for temp in TEMPERATURES:
        cursor.execute(
            "SELECT event_type, COUNT(*) FROM exp_translocation_events WHERE T_exp = ? GROUP BY event_type",
            (temp,),
        )
        results = dict(cursor.fetchall())
        success = results.get("success", 0)
        attempt = results.get("attempt", 0)

        cursor.execute("SELECT COUNT(*) FROM exp_worm_metrics WHERE T_exp = ?", (temp,))
        n_worms = cursor.fetchone()[0]

        print(f"  T={temp}°C: {success} success, {attempt} attempt, {n_worms} worms")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

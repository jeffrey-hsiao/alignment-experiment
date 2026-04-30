import sqlite3
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "pipelines" / "data" / "processed"
DB_PATH  = Path(__file__).parent / "alignment.db"

def load_jsonl(path, split, cursor):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cursor.execute(
                "INSERT INTO training_pairs (split, prompt, chosen, rejected) VALUES (?, ?, ?, ?)",
                (split, row["prompt"], row["chosen"], row["rejected"])
            )

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS training_pairs (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        split    TEXT,
        prompt   TEXT,
        chosen   TEXT,
        rejected TEXT
    )
""")
conn.execute("DELETE FROM training_pairs")

cur = conn.cursor()
load_jsonl(DATA_DIR / "train.jsonl", "train", cur)
load_jsonl(DATA_DIR / "val.jsonl",   "val",   cur)

conn.commit()

total = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
train = conn.execute("SELECT COUNT(*) FROM training_pairs WHERE split='train'").fetchone()[0]
val   = conn.execute("SELECT COUNT(*) FROM training_pairs WHERE split='val'").fetchone()[0]
conn.close()

print(f"完成！alignment.db 已建立")
print(f"  train：{train:,} 筆")
print(f"  val：  {val:,} 筆")
print(f"  合計： {total:,} 筆")

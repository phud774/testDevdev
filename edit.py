import json
import pickle
from pathlib import Path

json_path = Path(r"C:\coding_space\study\CS116\project\test_2025-12_submission.json")
pkl_path = json_path.with_suffix(".pkl")

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

with pkl_path.open("wb") as f:
    pickle.dump(data, f)

print(f"Saved to: {pkl_path}")
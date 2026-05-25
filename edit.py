import json
import pickle
from pathlib import Path

json_path = Path(r"C:\coding_space\study\CS116\project\submission_2026-01 (2).json")
pkl_path = json_path.with_suffix(".pkl")


def maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

data = {
    maybe_int(user_id): [str(item_id) for item_id in items]
    for user_id, items in data.items()
}

with pkl_path.open("wb") as f:
    pickle.dump(data, f)

print(f"Saved to: {pkl_path}")

with pkl_path.open("rb") as f:
    loaded_data = pickle.load(f)
print("\nPKL head:")
for idx, (user_id, items) in enumerate(loaded_data.items()):
    if idx >= 5:
        break
    print(type(items))

    print(f"{user_id!r} ({type(user_id).__name__}): {items[:10]}")

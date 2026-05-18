import polars as pl

# =========================================================
# FILE PATHS
# =========================================================

transaction_path = r"C:\coding_space\study\CS116\project\data\transaction_full_2025.parquet"
event_path = r"C:\coding_space\study\CS116\project\data\event_full_2025.parquet"

# =========================================================
# LOAD ONLY NEEDED COLUMNS
# =========================================================

transaction_df = pl.read_parquet(
    transaction_path,
    columns=["customer_id", "item_id"]
)

event_df = pl.read_parquet(
    event_path,
    columns=["customer_id", "item_id"]
)

# =========================================================
# CHECK SHAPE
# =========================================================

print("Transaction shape:", transaction_df.shape)
print("Event shape:", event_df.shape)

# =========================================================
# ROW-BY-ROW COMPARISON
# =========================================================

same_customer = (
    transaction_df["customer_id"] == event_df["customer_id"]
)

same_item = (
    transaction_df["item_id"] == event_df["item_id"]
)

same_pair = same_customer & same_item

# =========================================================
# RESULTS
# =========================================================

total_rows = len(same_pair)
matched_rows = same_pair.sum()
different_rows = total_rows - matched_rows

print("\n===== COMPARISON RESULT =====")
print(f"Total rows       : {total_rows:,}")
print(f"Matched rows     : {matched_rows:,}")
print(f"Different rows   : {different_rows:,}")

# =========================================================
# SHOW DIFFERENT ROWS
# =========================================================

if different_rows > 0:

    diff_indices = (
        pl.Series(range(total_rows))
        .filter(~same_pair)
        .to_list()
    )

    print("\n===== SAMPLE DIFFERENCES =====")

    sample_indices = diff_indices[:10]

    transaction_diff = transaction_df[sample_indices]
    event_diff = event_df[sample_indices]

    result = pl.DataFrame({
        "row_index": sample_indices,

        "transaction_customer_id":
            transaction_diff["customer_id"],

        "event_customer_id":
            event_diff["customer_id"],

        "transaction_item_id":
            transaction_diff["item_id"],

        "event_item_id":
            event_diff["item_id"],
    })

    print(result)

else:
    print("\nAll rows are identical!")
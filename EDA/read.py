import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

ROOT_DIR = r"C:\coding_space\study\CS116\project\data"

# Folder chứa các markdown report
OUTPUT_DIR = Path("parquet_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_bytes(size):
    power = 1024
    units = ["B", "KB", "MB", "GB", "TB"]

    n = 0
    while size >= power and n < len(units) - 1:
        size /= power
        n += 1

    return f"{size:.2f} {units[n]}"


def safe_sample(series, n=5):
    try:
        vals = series.dropna().astype(str).unique()[:n]
        return ", ".join(vals)
    except:
        return "N/A"


def markdown_table(headers, rows):
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in rows:
        md += "| " + " | ".join(map(str, row)) + " |\n"

    return md


def analyze_column(df, col):
    s = df[col]

    info = {
        "dtype": str(s.dtype),
        "null_count": int(s.isnull().sum()),
        "null_ratio": round(float(s.isnull().mean()), 6),
        "unique_count": int(s.nunique(dropna=True)),
    }

    try:
        info["memory_usage"] = format_bytes(s.memory_usage(deep=True))
    except:
        info["memory_usage"] = "N/A"

    # Numeric
    if pd.api.types.is_numeric_dtype(s):

        try:
            info.update({
                "min": s.min(),
                "max": s.max(),
                "mean": round(float(s.mean()), 6),
                "std": round(float(s.std()), 6),
                "median": round(float(s.median()), 6),
            })
        except:
            pass

    # Datetime
    elif pd.api.types.is_datetime64_any_dtype(s):

        try:
            info.update({
                "min_date": str(s.min()),
                "max_date": str(s.max()),
            })
        except:
            pass

    # Object/Text
    else:

        try:
            lengths = s.dropna().astype(str).map(len)

            info.update({
                "avg_length": round(float(lengths.mean()), 2) if len(lengths) else 0,
                "max_length": int(lengths.max()) if len(lengths) else 0,
                "sample_values": safe_sample(s)
            })

        except:
            pass

    return info


# =========================================================
# FIND PARQUET FILES
# =========================================================

parquet_files = list(Path(ROOT_DIR).rglob("*.parquet"))

print(f"Found {len(parquet_files)} parquet files")

# =========================================================
# PROCESS EACH FILE
# =========================================================

for idx, file in enumerate(parquet_files, 1):

    print(f"[{idx}/{len(parquet_files)}] Processing: {file.name}")

    report = []

    report.append(f"# Parquet Analysis Report\n")
    report.append(f"Generated at: `{datetime.now()}`\n")

    # =====================================================
    # FILE INFO
    # =====================================================

    stat = file.stat()

    report.append("## File Information\n")

    file_info_rows = [
        ["Filename", file.name],
        ["Full Path", str(file)],
        ["Size", format_bytes(stat.st_size)],
        ["Created", datetime.fromtimestamp(stat.st_ctime)],
        ["Modified", datetime.fromtimestamp(stat.st_mtime)],
    ]

    report.append(markdown_table(
        ["Property", "Value"],
        file_info_rows
    ))

    # =====================================================
    # PARQUET METADATA
    # =====================================================

    try:

        parquet_obj = pq.ParquetFile(file)

        report.append("\n## Parquet Metadata\n")

        metadata_rows = [
            ["Rows", parquet_obj.metadata.num_rows],
            ["Columns", parquet_obj.metadata.num_columns],
            ["Row Groups", parquet_obj.metadata.num_row_groups],
            ["Format Version", parquet_obj.metadata.format_version],
            ["Created By", parquet_obj.metadata.created_by],
        ]

        report.append(markdown_table(
            ["Metadata", "Value"],
            metadata_rows
        ))

        # =================================================
        # SCHEMA
        # =================================================

        report.append("\n## Schema\n")

        schema_rows = []

        for field in parquet_obj.schema_arrow:

            schema_rows.append([
                field.name,
                str(field.type),
                field.nullable
            ])

        report.append(markdown_table(
            ["Column", "Type", "Nullable"],
            schema_rows
        ))

    except Exception as e:

        report.append(f"\nMetadata extraction failed:\n```{e}```")

    # =====================================================
    # LOAD DATAFRAME
    # =====================================================

    try:

        df = pd.read_parquet(file)

        # =================================================
        # DATASET OVERVIEW
        # =================================================

        report.append("\n## Dataset Overview\n")

        overview_rows = [
            ["Shape", df.shape],
            ["Total Missing Values", int(df.isnull().sum().sum())],
            ["Duplicated Rows", int(df.duplicated().sum())],
            ["Memory Usage", format_bytes(df.memory_usage(deep=True).sum())],
        ]

        report.append(markdown_table(
            ["Metric", "Value"],
            overview_rows
        ))

        # =================================================
        # DTYPE DISTRIBUTION
        # =================================================

        report.append("\n## Data Type Distribution\n")

        dtype_counter = Counter(map(str, df.dtypes))

        dtype_rows = [
            [dtype, count]
            for dtype, count in dtype_counter.items()
        ]

        report.append(markdown_table(
            ["Dtype", "Count"],
            dtype_rows
        ))

        # =================================================
        # SAMPLE DATA
        # =================================================

        report.append("\n## Sample Rows\n")

        try:
            report.append(df.head(10).to_markdown(index=False))
        except:
            report.append("Cannot render preview.")

        # =================================================
        # COLUMN ANALYSIS
        # =================================================

        report.append("\n# Detailed Column Analysis\n")

        for col in df.columns:

            report.append(f"\n## Column: `{col}`\n")

            info = analyze_column(df, col)

            rows = [[k, v] for k, v in info.items()]

            report.append(markdown_table(
                ["Property", "Value"],
                rows
            ))

            # Top values
            try:

                vc = df[col].value_counts(dropna=False).head(10)

                vc_rows = [
                    [str(k), int(v)]
                    for k, v in vc.items()
                ]

                report.append("\n### Top Values\n")

                report.append(markdown_table(
                    ["Value", "Count"],
                    vc_rows
                ))

            except:
                pass

    except Exception as e:

        report.append(f"\nData loading failed:\n```{e}```")

    # =====================================================
    # SAVE MARKDOWN
    # =====================================================

    safe_name = file.stem.replace(" ", "_").replace("/", "_")
    output_path = OUTPUT_DIR / f"{safe_name}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, report)))

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\nDone.")
print(f"Reports saved in folder: {OUTPUT_DIR.resolve()}")
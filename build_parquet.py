#!/usr/bin/env python3
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# —— Inputs (adjust if paths differ)
CAPTIONS_CSV = "captions.csv"
META_CSV     = "case_metadata.csv"
GOLD_CSV     = "bboxes_gold.csv"
RATERS_CSV   = "bboxes_raters.csv"
IMAGES_DIR   = "images"

OUT_DIR      = "data"
OUT_PARQUET  = os.path.join(OUT_DIR, "nova-v1.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

# —— Load CSVs
captions = pd.read_csv(CAPTIONS_CSV)
meta     = pd.read_csv(META_CSV)
gold     = pd.read_csv(GOLD_CSV)
raters   = pd.read_csv(RATERS_CSV)

# —— Normalize key columns as strings
for df in (captions, meta, gold, raters):
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    if "scan_id" in df.columns:
        df["scan_id"] = df["scan_id"].astype(str)
    if "filename" in df.columns:
        df["filename"] = df["filename"].astype(str)

# —— Build the master list of images (from captions)
# If you prefer, you can take the union of filenames from all CSVs.
images = captions[["filename", "case_id", "scan_id", "caption"]].copy()
images["caption"] = images["caption"].fillna("").astype(str)

images.rename(columns={"caption": "caption_text"}, inplace=True)

# Add image_path and split
images["image_path"] = IMAGES_DIR + "/" + images["filename"]
images["split"] = "test"

# —— Prepare per-case metadata as a dict so we can merge by case_id
meta_cols = [
    "title", "publication_date", "clinical_history",
    "differential_diagnosis", "final_diagnosis", "link"
]
meta = meta[["case_id"] + meta_cols].copy()

# Merge per-image with per-case meta (left join by case_id)
df = images.merge(meta, on="case_id", how="left")

# —— Prepare gold bboxes
# Required columns: filename, x, y, width, height
gold_cols = ["filename", "x", "y", "width", "height"]
gold_use = gold[["filename", "x", "y", "width", "height"]].copy()
gold_use["source"] = "gold"
gold_use["label"] = ""  # no label provided; keep empty string

# —— Prepare rater bboxes
# Required columns: filename, rater, x, y, width, height
r_cols = ["filename", "rater", "x", "y", "width", "height"]
raters_use = raters[r_cols].copy()
raters_use.rename(columns={"rater": "source"}, inplace=True)
raters_use["label"] = ""

# —— Stack gold + raters, then group per filename into list[struct]
all_boxes = pd.concat([gold_use, raters_use], ignore_index=True)

def pack_boxes(group: pd.DataFrame):
    # Convert rows -> list of dicts with fixed keys
    return [
        {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "width": float(row["width"]),
            "height": float(row["height"]),
            "source": str(row["source"]),
            "label": str(row["label"]),
        }
        for _, row in group.iterrows()
    ]

boxes_per_file = (
    all_boxes
    .groupby("filename", sort=False, group_keys=False)
    .apply(pack_boxes)
    .rename("bboxes")
    .reset_index()
)

# Attach bboxes to df by filename; missing -> empty list
df = df.merge(boxes_per_file, on="filename", how="left")
df["bboxes"] = df["bboxes"].apply(lambda v: v if isinstance(v, list) else [])

# —— Build a nested Arrow schema for clean typing
bbox_struct = pa.struct([
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("width", pa.float64()),
    ("height", pa.float64()),
    ("source", pa.string()),
    ("label", pa.string()),
])

meta_struct = pa.struct([
    ("title", pa.string()),
    ("publication_date", pa.string()),  # keep as-is (e.g., 30.03.2022); can normalize later
    ("clinical_history", pa.string()),
    ("differential_diagnosis", pa.string()),
    ("final_diagnosis", pa.string()),
    ("link", pa.string()),
])

schema = pa.schema([
    ("image_path", pa.string()),
    ("filename", pa.string()),
    ("split", pa.string()),
    ("case_id", pa.string()),
    ("scan_id", pa.string()),
    ("caption_text", pa.string()),
    ("bboxes", pa.list_(bbox_struct)),
    ("meta", meta_struct),
])
for col in ["title", "publication_date", "clinical_history", "differential_diagnosis", "final_diagnosis", "link"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)
# —— Build Arrow arrays column by column
def to_arrow_array_list_of_struct(list_of_dicts, struct_type):
    # list_of_dicts: python list of dicts with keys matching struct_type
    # returns: pa.Array(list<struct>)
    if not list_of_dicts:
        # represent empty with empty list
        return pa.array([], type=pa.list_(struct_type))
    # we will build a ListArray from a chunk; handle per row later
    raise RuntimeError("This helper is row-wise; we will build below with pa.array on the full column.")

# Build meta struct column
meta_dicts = df[meta_cols].to_dict(orient="records")
meta_array = pa.array(meta_dicts, type=meta_struct)

# Build bboxes list<struct> column
# Flatten per row using pa.array with explicit type
bboxes_array = pa.array(df["bboxes"].tolist(), type=pa.list_(bbox_struct))

# Build the rest as Arrow arrays
table = pa.Table.from_arrays(
    [
        pa.array(df["image_path"].tolist(), type=pa.string()),
        pa.array(df["filename"].tolist(), type=pa.string()),
        pa.array(df["split"].tolist(), type=pa.string()),
        pa.array(df["case_id"].tolist(), type=pa.string()),
        pa.array(df["scan_id"].tolist(), type=pa.string()),
        pa.array(df["caption_text"].tolist(), type=pa.string()),
        bboxes_array,
        meta_array,
    ],
    schema=schema
)

# —— Optional: sanity checks
# - ensure files referenced actually exist
missing = [p for p in df["image_path"] if not os.path.exists(p)]
if missing:
    print(f"[WARN] {len(missing)} image paths do not exist locally. First few:", missing[:5])

# —— Write Parquet
pq.write_table(table, OUT_PARQUET)
print(f"Wrote {OUT_PARQUET} with {table.num_rows} rows.")


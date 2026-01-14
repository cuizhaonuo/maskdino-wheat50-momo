import os
import csv
from PIL import Image, ExifTags

# ROOT = "/public/home/zncui/datasets/maize_rice_dataset/maize100"
ROOT = "/public/lab_share/nfs_field/raw_data/DJI/M3M/wuhan2025/2025wuhan-rice/2025wuhan_yuexiang2/20250703"
OUT_CSV = "maize100_exif.csv"

# EXIF tag name mapping
TAGS = {v: k for k, v in ExifTags.TAGS.items()}
GPSTAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}

def _rational_to_float(x):
    # x might be (num, den) or PIL Rational
    try:
        return float(x)
    except Exception:
        pass
    try:
        return x[0] / x[1] if x[1] else None
    except Exception:
        return None

def parse_gps_altitude(gps):
    """
    gps is a dict with GPS tags decoded by PIL.
    Returns altitude (float) and ref (0/1 or 'Above/Below').
    """
    if not gps:
        return None, None
    alt = gps.get("GPSAltitude")
    ref = gps.get("GPSAltitudeRef")  # 0=Above sea level, 1=Below sea level (per EXIF)
    alt_f = _rational_to_float(alt) if alt is not None else None
    if ref is not None:
        try:
            ref = int(ref)
        except Exception:
            pass
    return alt_f, ref

def get_exif_dict(img):
    exif = img.getexif()
    if not exif:
        return {}
    out = {}
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        out[tag_name] = value
    # GPS is nested
    if "GPSInfo" in out and isinstance(out["GPSInfo"], dict):
        gps_raw = out["GPSInfo"]
        gps = {}
        for k, v in gps_raw.items():
            gps_name = ExifTags.GPSTAGS.get(k, k)
            gps[gps_name] = v
        out["GPSInfo"] = gps
    return out

def try_get_lens_info(exif):
    # Common EXIF fields
    lens_model = exif.get("LensModel")
    # Some cameras store lens info in "LensSpecification" (4 rationals: minF, maxF, minA, maxA)
    lens_spec = exif.get("LensSpecification")
    focal = exif.get("FocalLength")
    focal_35 = exif.get("FocalLengthIn35mmFilm")

    focal = _rational_to_float(focal) if focal is not None else None
    try:
        focal_35 = int(focal_35) if focal_35 is not None else None
    except Exception:
        pass

    return lens_model, lens_spec, focal, focal_35

def iter_jpgs(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg"):
            # if fn.endswith(".JPG"):
                yield os.path.join(dirpath, fn)

rows = []
for path in iter_jpgs(ROOT):
    try:
        with Image.open(path) as img:
            exif = get_exif_dict(img)

        make = exif.get("Make")
        model = exif.get("Model")
        dt = exif.get("DateTimeOriginal") or exif.get("DateTime")

        gps = exif.get("GPSInfo")
        gps_alt, gps_alt_ref = parse_gps_altitude(gps)

        lens_model, lens_spec, focal, focal_35 = try_get_lens_info(exif)

        rows.append({
            "file_path": path,
            "make": make,
            "model": model,
            "datetime_original": dt,
            "gps_altitude": gps_alt,
            "gps_altitude_ref": gps_alt_ref,
            "lens_model": lens_model,
            "lens_specification": str(lens_spec) if lens_spec is not None else None,
            "focal_length_mm": focal,
            "focal_length_35mm": focal_35,
        })

    except Exception as e:
        rows.append({
            "file_path": path,
            "error": str(e),
        })

fieldnames = [
    "file_path", "make", "model", "datetime_original",
    "gps_altitude", "gps_altitude_ref",
    "lens_model", "lens_specification",
    "focal_length_mm", "focal_length_35mm",
    "error"
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        # Ensure all keys exist
        for k in fieldnames:
            r.setdefault(k, None)
        w.writerow(r)

print(f"Saved: {OUT_CSV}  (rows={len(rows)})")

"""
Converts all sub{1-97}_{1-4}.hdf5 files to .set format.

Output format choices (set FORMAT below):
  "edf"    → .edf files
  "set"    → .set files 
  
Output folder structure:
  OUTPUT_DIR/
    HC/
      sub1_1.edf
      sub1_2.edf
      ...
    MCI/
      sub2_1.edf
      ...
    SCD/
      ...
    FLAGGED/          ← sub93 and other problematic files
      sub93_1.edf
      ...
    conversion_log.csv

Requirements:
    pip install h5py pandas openpyxl numpy mne
"""

import h5py
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import mne
import logging, sys
from pathlib import Path
from collections import defaultdict

# make stdout/stderr UTF-8 so non-ASCII status chars don't crash on cp950/other consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

mne.set_log_level("WARNING")   # suppress MNE's verbose output

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR = r"D:\2026\Prof hsu project\Dementia_dataset\datasets"
GROUP_XLSX  = r"D:\2026\Prof hsu project\Dementia_dataset\datasets\nthu_group.xlsx"
OUTPUT_DIR  = str(Path(__file__).resolve().parents[1] / "data" / "converted")   # -> <repo>/data/converted

FORMAT      ="set"    # "edf" or "set"
SFREQ       = 250.0    # fallback if not found in file

# Canonical 32-channel layout (uppercase for matching)
CANONICAL_UPPER = [
    'FP1', 'FP2', 'AF3', 'AF4', 'F7',  'F3',  'FZ',  'F4',  'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7',  'C3',  'CZ',  'C4',  'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7',  'P3',  'PZ',  'P4',  'P8',
    'PO7', 'PO3', 'PO4', 'PO8', 'OZ',
]

# Display names (mixed-case, used in MNE/EDF)
CANONICAL_DISPLAY = [
    'Fp1', 'Fp2', 'AF3', 'AF4', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7',  'C3',  'Cz',  'C4',  'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7',  'P3',  'Pz',  'P4',  'P8',
    'PO7', 'PO3', 'PO4', 'PO8', 'Oz',
]

# Physical channel number to canonical name for unlabeled files

PHYS_TO_CANONICAL = {i+1: CANONICAL_DISPLAY[i] for i in range(32)}

CANONICAL_SET = set(CANONICAL_UPPER)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# XML PARSER  (reused from analyzer)
# ─────────────────────────────────────────────

def parse_acquisition_xml(xml_bytes):
    """
    Returns:
        channel_info : list of dicts {name, type, logical, physical}
                       — in file order, all names normalised to uppercase
        sfreq        : float
        filter_info  : dict {highpass_hz, lowpass_hz, notch_hz}
        unlabeled    : bool   , True if channels have numeric-only names
    """
    if isinstance(xml_bytes, (bytes, np.bytes_)):
        xml_str = xml_bytes.decode("utf-8", errors="replace")
    else:
        xml_str = str(xml_bytes)

    root        = ET.fromstring(xml_str)
    channel_info = []
    filter_info  = {}
    sfreq        = None
    unlabeled    = False

    sf_el = root.find("SamplingFrequency")
    if sf_el is not None and sf_el.text:
        try:
            sfreq = float(sf_el.text)
        except ValueError:
            pass

    for ch_block in root.iter("ChannelProperties"):
        name_el  = ch_block.find("ChannelName")
        type_el  = ch_block.find("ChannelType")
        log_el   = ch_block.find("LogicalChannelNumber")
        phys_el  = ch_block.find("PhysicalChannelNumber")
        sr_el    = ch_block.find("SampleRate")
        hp_el    = ch_block.find("HighpassFilter")
        lp_el    = ch_block.find("LowpassFilter")
        notch_el = ch_block.find("NotchFilter")

        if name_el is None or type_el is None:
            continue

        ch_name  = name_el.text.strip()
        ch_type  = type_el.text.strip().upper()
        logical  = int(log_el.text)  if log_el  is not None else -1
        physical = int(phys_el.text) if phys_el is not None else -1

        if sfreq is None and sr_el is not None:
            try:
                sfreq = float(sr_el.text)
            except (ValueError, TypeError):
                pass

        if not filter_info and ch_type == "EEG":
            try:
                hp    = float(hp_el.text)    if hp_el    is not None else None
                lp    = float(lp_el.text)    if lp_el    is not None else None
                notch = float(notch_el.text) if notch_el is not None else None
                # -1 means filter disabled
                filter_info = {
                    "highpass_hz": hp    if (hp    is not None and hp    >= 0) else None,
                    "lowpass_hz" : lp    if (lp    is not None and lp    >= 0) else None,
                    "notch_hz"   : notch if (notch is not None and notch >= 0) else None,
                }
            except (ValueError, TypeError):
                pass

        # detect numeric-only names
        if ch_name.isdigit():
            unlabeled = True

        channel_info.append({
            "name"    : ch_name,
            "upper"   : ch_name.upper(),
            "type"    : ch_type,
            "logical" : logical,
            "physical": physical,
        })

    return channel_info, sfreq or SFREQ, filter_info, unlabeled


# ─────────────────────────────────────────────
# CHANNEL RESOLVER
# ─────────────────────────────────────────────

def resolve_eeg_channels(channel_info, unlabeled, filepath_str):
    """
    Returns:
        col_indices   : list[int]  - column indices into RawData/Samples to keep
        display_names : list[str]  - corresponding display channel names
        warnings      : list[str]  - any issues found
        flagged       : bool       - True if file needs manual review
    """
    warnings = []
    flagged  = False

    if unlabeled:
        flagged = True
        warnings.append("Channel names are numeric - remapped via physical channel number.")
        warnings.append("Filters were disabled at recording time - data is unfiltered.")
        col_indices   = []
        display_names = []
        for i, ch in enumerate(channel_info):
            phys = ch["physical"]
            if phys in PHYS_TO_CANONICAL:
                col_indices.append(i)
                display_names.append(PHYS_TO_CANONICAL[phys])
            # skip anything without a canonical mapping
        return col_indices, display_names, warnings, flagged

    # Normal labelled files
    name_to_col = {}
    for i, ch in enumerate(channel_info):
        upper = ch["upper"]
        ctype = ch["type"]

        # Treat EOG as EEG if it's a canonical electrode
        if ctype == "EOG" and upper in CANONICAL_SET:
            warnings.append(f"Channel {ch['name']} is labeled EOG but is a canonical EEG electrode — included as EEG.")
            name_to_col[upper] = i
        elif ctype == "EEG" and upper in CANONICAL_SET:
            name_to_col[upper] = i
        # everything else is skipped

    # Build output in canonical order
    col_indices   = []
    display_names = []
    missing       = []
    for upper, display in zip(CANONICAL_UPPER, CANONICAL_DISPLAY):
        if upper in name_to_col:
            col_indices.append(name_to_col[upper])
            display_names.append(display)
        else:
            missing.append(display)

    if missing:
        flagged = True
        warnings.append(f"Missing canonical channels: {missing}")

    return col_indices, display_names, warnings, flagged


# ─────────────────────────────────────────────
# SINGLE FILE CONVERTER
# ─────────────────────────────────────────────

def convert_file(hdf5_path, out_path, group, sub_id, seg):
    """
    Convert one HDF5 file to EDF or SET.
    Returns a log dict.
    """
    entry = {
        "file"    : hdf5_path.name,
        "sub_id"  : sub_id,
        "seg"     : seg,
        "group"   : group,
        "status"  : "ok",
        "warnings": "",
        "out_file": "",
    }

    try:
        with h5py.File(hdf5_path, "r") as f:
            raw_data     = f["RawData/Samples"][:]          # (n_samples, n_ch)
            xml_raw      = f["RawData/AcquisitionTaskDescription"][0]

            # Events
            ev_times  = f["AsynchronData/Time"][:].flatten()   if "AsynchronData/Time"   in f else np.array([])
            ev_typeids = f["AsynchronData/TypeID"][:].flatten() if "AsynchronData/TypeID" in f else np.array([])
            ev_values  = f["AsynchronData/Value"][:].flatten()  if "AsynchronData/Value"  in f else np.array([])

        channel_info, sfreq, filter_info, unlabeled = parse_acquisition_xml(xml_raw)
        col_indices, display_names, warnings, flagged = resolve_eeg_channels(
            channel_info, unlabeled, str(hdf5_path)
        )

        if not col_indices:
            entry["status"]   = "error"
            entry["warnings"] = "No EEG channels could be resolved."
            return entry, flagged

        # Extract and transpose: (n_ch, n_samples), convert µV → V
        eeg_data = raw_data[:, col_indices].T.astype(np.float64) * 1e-6

        # Build MNE Raw object
        info = mne.create_info(
            ch_names=display_names,
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        # Add standard 10-20 montage for channels that exist in it
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose=False)

        # Add events from AsynchronData
        if len(ev_times) > 0:
            # Times are in samples (int64); convert to MNE annotations
            onset_sec  = ev_times.astype(float) / sfreq
            durations  = np.zeros(len(onset_sec))
            # Use Value as description if available, else TypeID
            if len(ev_values) == len(onset_sec):
                descriptions = [f"val_{int(v)}" for v in ev_values]
            elif len(ev_typeids) == len(onset_sec):
                descriptions = [f"type_{int(t)}" for t in ev_typeids]
            else:
                descriptions = ["event"] * len(onset_sec)
            annotations = mne.Annotations(onset_sec, durations, descriptions)
            raw.set_annotations(annotations, verbose=False)

        # Export
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if FORMAT == "edf":
            raw.export(str(out_path), fmt="edf", overwrite=True, verbose=False)
        else:
            raw.export(str(out_path), fmt="eeglab", overwrite=True, verbose=False)

        entry["status"]   = "flagged" if flagged else "ok"
        entry["warnings"] = " | ".join(warnings) if warnings else ""
        entry["out_file"] = str(out_path)

    except Exception as e:
        entry["status"]   = "error"
        entry["warnings"] = str(e)

    return entry, flagged


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("═" * 72)
    print("  BATCH HDF5 → " + FORMAT.upper() + " CONVERTER")
    print("═" * 72)

    # ── Load group labels ────────────────────────────────────────────
    df = pd.read_excel(GROUP_XLSX)
    df.columns = [c.strip() for c in df.columns]
    sub_col   = next(c for c in df.columns if c.lower() in ("sub", "subject", "id"))
    group_col = next(c for c in df.columns if c.lower() in ("group", "label", "class"))
    df[sub_col] = df[sub_col].astype(int)
    sub_to_group = dict(zip(df[sub_col], df[group_col].str.strip().str.upper()))

    # ── Find files ───────────────────────────────────────────────────
    dataset_path = Path(DATASET_DIR)
    output_path  = Path(OUTPUT_DIR)
    hdf5_files   = sorted(dataset_path.glob("sub*.hdf5"))
    ext          = f".{FORMAT}"

    print(f"\n  Input  : {DATASET_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Format : {FORMAT.upper()}")
    print(f"  Files  : {len(hdf5_files)} found\n")

    # ── Convert ──────────────────────────────────────────────────────
    log_entries = []
    counts = defaultdict(int)

    for i, fp in enumerate(hdf5_files, 1):
        stem  = fp.stem
        parts = stem.split("_")
        try:
            sub_id = int(parts[0].replace("sub", ""))
            seg    = int(parts[1])
        except (ValueError, IndexError):
            log.warning(f"  [SKIP] Cannot parse: {fp.name}")
            continue

        group    = sub_to_group.get(sub_id, "UNKNOWN")
        out_dir  = output_path / ("FLAGGED" if sub_id == 93 else group)
        out_file = out_dir / f"sub{sub_id}_{seg}{ext}"

        print(f"  [{i:>3}/{len(hdf5_files)}] {fp.name:<22} → {group}", end="  ", flush=True)

        entry, flagged = convert_file(fp, out_file, group, sub_id, seg)

        if flagged and group != "FLAGGED":
            # move to FLAGGED folder
            flagged_out = output_path / "FLAGGED" / f"sub{sub_id}_{seg}{ext}"
            flagged_out.parent.mkdir(parents=True, exist_ok=True)
            if out_file.exists():
                out_file.rename(flagged_out)
            entry["out_file"] = str(flagged_out)

        status = entry["status"]
        counts[status] += 1

        if status == "ok":
            print("✓")
        elif status == "flagged":
            print(f"FLAGGED — {entry['warnings']}")
        else:
            print(f"ERROR — {entry['warnings']}")

        log_entries.append(entry)

    # ── Save log ─────────────────────────────────────────────────────
    log_path = output_path / "conversion_log.csv"
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_entries).to_csv(log_path, index=False)

    print("  DONE")



if __name__ == "__main__":
    main()

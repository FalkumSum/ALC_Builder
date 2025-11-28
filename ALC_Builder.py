import streamlit as st


# ---------- Parsing helpers ----------

def parse_header_line(raw: str):
    """
    Parse the XYZ header line (starting with /) into:
    - cols: list of column names
    - pos: dict of column_name -> 1-based index
    """
    line = raw.strip()
    if line.startswith("/"):
        line = line[1:]
    cols = line.split()
    pos = {name: i + 1 for i, name in enumerate(cols)}  # 1-based indices
    return cols, pos


def parse_first_data_line(raw: str):
    """
    Parse the first data line into a list of values, split on whitespace.
    """
    return raw.strip().split()


def find_indexed_columns_with_prefix(cols, prefix):
    """
    Find columns like prefix[0], prefix[1], ...
    Example:
        prefix = "LM_Z_dBdt"
        matches "LM_Z_dBdt[0]", "LM_Z_dBdt[1]", ...
    Returns list of column names sorted by the index in [idx].
    """
    found = []
    for name in cols:
        if name.startswith(prefix) and "[" in name and "]" in name:
            try:
                idx_str = name.split("[", 1)[1].split("]", 1)[0]
                gate_index = int(idx_str)
                found.append((gate_index, name))
            except ValueError:
                continue
    found_sorted = [n for _, n in sorted(found)]
    return found_sorted


def get_first_non_empty_indexed(cols, prefixes):
    """
    Try a list of prefixes in order, return the first non-empty list of
    indexed columns (using find_indexed_columns_with_prefix).
    """
    for p in prefixes:
        lst = find_indexed_columns_with_prefix(cols, p)
        if lst:
            return lst
    return []


def find_current_index(pos, label):
    """
    Find current column index for LM or HM, case-insensitive.
    label: 'LM' or 'HM'
    """
    candidates = []
    if label == "LM":
        candidates = ["LMcurrent", "LMCurrent"]
    elif label == "HM":
        candidates = ["HMcurrent", "HMCurrent"]

    for cand in candidates:
        for name, idx in pos.items():
            if name.lower() == cand.lower():
                return idx

    return -1


# ---------- ALC builder ----------

def build_alc_text(
    cols,
    pos,
    system_name="SkyTEM XYZ",
    ch1_label="LM",
    ch2_label="HM",
):
    """
    Build ALC text given parsed header and mapping.
    ch1_label and ch2_label are "LM" or "HM".
    Returns: (alc_text, info, layout)
    """

    def col_or_minus1(name):
        return pos.get(name, -1)

    # -------- Gate columns --------
    # LM gates: LM_Z_dBdt[0..]
    lm_gates = get_first_non_empty_indexed(cols, ["LM_Z_dBdt"])

    # HM gates: HM_Z_dBdt[0..] or HM_Z_dbdt[0..] (handle both variants)
    hm_gates = get_first_non_empty_indexed(cols, ["HM_Z_dBdt", "HM_Z_dbdt"])

    channels_present = []
    if lm_gates:
        channels_present.append("LM")
    if hm_gates:
        channels_present.append("HM")

    if not channels_present:
        raise ValueError("No LM/HM gate columns detected in header.")

    # Resolve channel roles (Ch01 / Ch02) according to user choice, but fallback if needed
    if ch1_label not in channels_present:
        ch1_label = channels_present[0]
    if ch2_label not in channels_present or ch2_label == ch1_label:
        ch2_label = None

    channels_number = 1 if ch2_label is None else 2

    # -------- STD columns (relative uncertainties) --------
    # We support multiple name patterns:
    #  - Older: RelUnc_LM_Z_dBdt_Merge, RelUnc_HM_Z_dBdt
    #  - Newer: RelUnc_SWch1_G01, RelUnc_SWch2_G01
    def std_for_label(label):
        if label == "LM":
            prefixes = [
                "RelUnc_LM_Z_dBdt",
                "RelUnc_LM",
                "RelUnc_SWch1_G01",
                "RelUnc_SWch1",
            ]
        else:  # HM
            prefixes = [
                "RelUnc_HM_Z_dBdt",
                "RelUnc_HM",
                "RelUnc_SWch2_G01",
                "RelUnc_SWch2",
            ]
        return get_first_non_empty_indexed(cols, prefixes)

    def gates_for_label(label):
        return lm_gates if label == "LM" else hm_gates

    # Gate + STD lists for Ch01 and Ch02
    gates_ch1 = gates_for_label(ch1_label)
    std_ch1 = std_for_label(ch1_label)
    current_ch1 = find_current_index(pos, ch1_label)

    if ch2_label:
        gates_ch2 = gates_for_label(ch2_label)
        std_ch2 = std_for_label(ch2_label)
        current_ch2 = find_current_index(pos, ch2_label)
    else:
        gates_ch2, std_ch2, current_ch2 = [], [], -1

    # ----- Build lines -----
    def kv(key, val):
        return f"{key:<22}= {val}"

    lines_out = []

    # Header / fixed fields
    lines_out.append(kv("Version", 2))
    lines_out.append(kv("System", system_name))
    lines_out.append(kv("ChannelsNumber", channels_number))
    lines_out.append(kv("Date", col_or_minus1("Date")))
    lines_out.append(kv("Dummy", "*"))
    lines_out.append(kv("Line", col_or_minus1("Line")))
    lines_out.append(kv("Magnetic", col_or_minus1("TMI")))
    lines_out.append(kv("Misc1", -1))
    lines_out.append(kv("Misc2", -1))
    lines_out.append(kv("Misc3", -1))
    lines_out.append(kv("Misc4", -1))
    lines_out.append(kv("RxPitch", -1))
    lines_out.append(kv("RxRoll", -1))
    lines_out.append(kv("Time", col_or_minus1("Time")))
    lines_out.append(kv("Topography", col_or_minus1("DEM")))
    lines_out.append(kv("TxAltitude", col_or_minus1("Height")))
    lines_out.append(kv("TxOffTime", -1))
    lines_out.append(kv("TxOnTime", -1))
    lines_out.append(kv("TxPeakTime", -1))
    lines_out.append(kv("TxPitch", col_or_minus1("AngleX")))
    lines_out.append(kv("TxRoll", col_or_minus1("AngleY")))
    lines_out.append(kv("TxRxHoriSep", -1))
    lines_out.append(kv("TxRxVertSep", -1))
    lines_out.append(kv("UTMX", col_or_minus1("E")))
    lines_out.append(kv("UTMY", col_or_minus1("N")))
    lines_out.append(kv("Current_Ch01", current_ch1))
    if channels_number == 2:
        lines_out.append(kv("Current_Ch02", current_ch2))
    lines_out.append(kv("PowerLineMonitor", col_or_minus1("PLNI")))
    lines_out.append("")  # blank line

    # Gates Ch01
    for i, name in enumerate(gates_ch1, start=1):
        lines_out.append(kv(f"Gate_Ch01_{i:02d}", pos[name]))
    lines_out.append("")

    # Gates Ch02
    if channels_number == 2:
        for i, name in enumerate(gates_ch2, start=1):
            lines_out.append(kv(f"Gate_Ch02_{i:02d}", pos[name]))
        lines_out.append("")

    # STD Ch01
    for i, name in enumerate(std_ch1, start=1):
        lines_out.append(kv(f"STD_Ch01_{i:02d}", pos[name]))
    lines_out.append("")

    # STD Ch02
    if channels_number == 2:
        for i, name in enumerate(std_ch2, start=1):
            lines_out.append(kv(f"STD_Ch02_{i:02d}", pos[name]))
        lines_out.append("")

    # InUse – all -1
    for i in range(1, len(gates_ch1) + 1):
        lines_out.append(kv(f"InUse_Ch01_{i:02d}", -1))
    lines_out.append("")

    if channels_number == 2:
        for i in range(1, len(gates_ch2) + 1):
            lines_out.append(kv(f"InUse_Ch02_{i:02d}", -1))

    info = {
        "channels_number": channels_number,
        "ch1_label": ch1_label,
        "ch2_label": ch2_label,
        "n_gates_ch1": len(gates_ch1),
        "n_gates_ch2": len(gates_ch2),
        "n_std_ch1": len(std_ch1),
        "n_std_ch2": len(std_ch2),
    }

    layout = {
        "ch1_label": ch1_label,
        "ch2_label": ch2_label,
        "gates_ch1": gates_ch1,
        "gates_ch2": gates_ch2,
        "std_ch1": std_ch1,
        "std_ch2": std_ch2,
        "current_ch1": current_ch1,
        "current_ch2": current_ch2,
        "field_indices": {
            "Line": col_or_minus1("Line"),
            "Date": col_or_minus1("Date"),
            "Time": col_or_minus1("Time"),
            "TxPitch": col_or_minus1("AngleX"),
            "TxRoll": col_or_minus1("AngleY"),
            "TxAltitude": col_or_minus1("Height"),
            "UTMX": col_or_minus1("E"),
            "UTMY": col_or_minus1("N"),
            "Topography": col_or_minus1("DEM"),
            "Magnetic": col_or_minus1("TMI"),
            "PowerLineMonitor": col_or_minus1("PLNI"),
        },
    }

    return "\n".join(lines_out), info, layout


# ---------- 3-row mapping view helpers ----------

def build_core_mapping(layout, pos, first_values):
    rows = []

    fi = layout["field_indices"]

    def add_row(alc_name, xyz_name, xyz_index):
        if xyz_index <= 0:
            first_val = ""
        else:
            first_val = first_values[xyz_index - 1] if len(first_values) >= xyz_index else ""
        rows.append(
            {
                "ALC entry": f"{alc_name} = {xyz_index}",
                "XYZ column": f"{xyz_index} → {xyz_name}",
                "First value": first_val,
            }
        )

    add_row("Line", "Line", fi["Line"])
    add_row("Date", "Date", fi["Date"])
    add_row("Time", "Time", fi["Time"])
    add_row("TxPitch", "AngleX", fi["TxPitch"])
    add_row("TxRoll", "AngleY", fi["TxRoll"])
    add_row("TxAltitude", "Height", fi["TxAltitude"])
    add_row("UTMX", "E", fi["UTMX"])
    add_row("UTMY", "N", fi["UTMY"])
    add_row("Topography", "DEM", fi["Topography"])
    add_row("Magnetic", "TMI", fi["Magnetic"])
    add_row("PowerLineMonitor", "PLNI", fi["PowerLineMonitor"])

    # Currents: find the column name from pos by index
    def find_name_by_index(idx):
        for name, i in pos.items():
            if i == idx:
                return name
        return "?"

    idx_ch1 = layout["current_ch1"]
    if idx_ch1 > 0:
        xyz_name_ch1 = find_name_by_index(idx_ch1)
        first_val = first_values[idx_ch1 - 1] if len(first_values) >= idx_ch1 else ""
        rows.append(
            {
                "ALC entry": f"Current_Ch01 = {idx_ch1}",
                "XYZ column": f"{idx_ch1} → {xyz_name_ch1}",
                "First value": first_val,
            }
        )

    idx_ch2 = layout["current_ch2"]
    if idx_ch2 > 0:
        xyz_name_ch2 = find_name_by_index(idx_ch2)
        first_val = first_values[idx_ch2 - 1] if len(first_values) >= idx_ch2 else ""
        rows.append(
            {
                "ALC entry": f"Current_Ch02 = {idx_ch2}",
                "XYZ column": f"{idx_ch2} → {xyz_name_ch2}",
                "First value": first_val,
            }
        )

    return rows


de

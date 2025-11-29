import streamlit as st
import pandas as pd


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
    overrides=None,
):
    """
    Build ALC text given parsed header and mapping.
    ch1_label and ch2_label are "LM" or "HM".
    overrides: dict mapping ALC field name -> header name (from XYZ) to override index.
    Returns: (alc_text, info, layout)
    """
    overrides = overrides or {}

    def idx_from_header(header_name: str) -> int:
        if not header_name:
            return -1
        return pos.get(header_name, -1)

    def col_or_minus1(field_name, default_header_name):
        """
        field_name: ALC field name, e.g. 'Line', 'Date', 'Misc1', ...
        default_header_name: the header name we try by default, e.g. 'Line', 'Date', etc.
        If overrides[field_name] exists, we use that header instead.
        """
        header_to_use = overrides.get(field_name, default_header_name)
        if header_to_use is None:
            return -1
        return idx_from_header(header_to_use)

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

    # Resolve channel roles (Ch01 / Ch02) according to user choice, but fallback if needed.
    # This also covers HM-only or LM-only cases.
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

    # Currents allow overrides: 'Current_Ch01', 'Current_Ch02'
    if ch1_label == "LM":
        default_ch1_header = "LMcurrent"
    else:
        default_ch1_header = "HMcurrent"
    current_ch1 = idx_from_header(overrides.get("Current_Ch01", default_ch1_header))

    if ch2_label:
        if ch2_label == "LM":
            default_ch2_header = "LMcurrent"
        else:
            default_ch2_header = "HMcurrent"
        current_ch2 = idx_from_header(overrides.get("Current_Ch02", default_ch2_header))
        gates_ch2 = gates_for_label(ch2_label)
        std_ch2 = std_for_label(ch2_label)
    else:
        current_ch2 = 0  # 0 = we don't expect Ch02
        gates_ch2, std_ch2 = [], []

    # ----- Build lines -----
    def kv(key, val):
        return f"{key:<22}= {val}"

    lines_out = []

    # Header / fixed fields (with optional overrides)
    idx_date = col_or_minus1("Date", "Date")
    idx_line = col_or_minus1("Line", "Line")
    idx_time = col_or_minus1("Time", "Time")
    idx_anglex = col_or_minus1("TxPitch", "AngleX")
    idx_angley = col_or_minus1("TxRoll", "AngleY")
    idx_height = col_or_minus1("TxAltitude", "Height")
    idx_e = col_or_minus1("UTMX", "E")
    idx_n = col_or_minus1("UTMY", "N")
    idx_dem = col_or_minus1("Topography", "DEM")
    idx_tmi = col_or_minus1("Magnetic", "TMI")
    idx_plni = col_or_minus1("PowerLineMonitor", "PLNI")

    # Misc fields are typically unassigned by default (-1), but user can override.
    idx_misc1 = col_or_minus1("Misc1", None)
    idx_misc2 = col_or_minus1("Misc2", None)
    idx_misc3 = col_or_minus1("Misc3", None)
    idx_misc4 = col_or_minus1("Misc4", None)

    # TxRx separations (allow overrides, default None)
    idx_txrx_h = col_or_minus1("TxRxHoriSep", None)
    idx_txrx_v = col_or_minus1("TxRxVertSep", None)

    lines_out.append(kv("Version", 2))
    lines_out.append(kv("System", system_name))
    lines_out.append(kv("ChannelsNumber", channels_number))
    lines_out.append(kv("Date", idx_date))
    lines_out.append(kv("Dummy", "*"))
    lines_out.append(kv("Line", idx_line))
    lines_out.append(kv("Magnetic", idx_tmi))
    lines_out.append(kv("Misc1", idx_misc1 if idx_misc1 != -1 else -1))
    lines_out.append(kv("Misc2", idx_misc2 if idx_misc2 != -1 else -1))
    lines_out.append(kv("Misc3", idx_misc3 if idx_misc3 != -1 else -1))
    lines_out.append(kv("Misc4", idx_misc4 if idx_misc4 != -1 else -1))
    lines_out.append(kv("RxPitch", -1))
    lines_out.append(kv("RxRoll", -1))
    lines_out.append(kv("Time", idx_time))
    lines_out.append(kv("Topography", idx_dem))
    lines_out.append(kv("TxAltitude", idx_height))
    lines_out.append(kv("TxOffTime", -1))
    lines_out.append(kv("TxOnTime", -1))
    lines_out.append(kv("TxPeakTime", -1))
    lines_out.append(kv("TxPitch", idx_anglex))
    lines_out.append(kv("TxRoll", idx_angley))
    lines_out.append(kv("TxRxHoriSep", idx_txrx_h if idx_txrx_h != -1 else -1))
    lines_out.append(kv("TxRxVertSep", idx_txrx_v if idx_txrx_v != -1 else -1))
    lines_out.append(kv("UTMX", idx_e))
    lines_out.append(kv("UTMY", idx_n))
    lines_out.append(kv("Current_Ch01", current_ch1 if current_ch1 > 0 else -1))
    if channels_number == 2:
        lines_out.append(kv("Current_Ch02", current_ch2 if current_ch2 > 0 else -1))
    lines_out.append(kv("PowerLineMonitor", idx_plni))
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

    field_indices = {
        "Line": idx_line,
        "Date": idx_date,
        "Time": idx_time,
        "Topography": idx_dem,
        "TxAltitude": idx_height,
        "TxPitch": idx_anglex,
        "TxRoll": idx_angley,
        "UTMX": idx_e,
        "UTMY": idx_n,
        "Magnetic": idx_tmi,
        "PowerLineMonitor": idx_plni,
        "Misc1": idx_misc1 if idx_misc1 != -1 else -1,
        "Misc2": idx_misc2 if idx_misc2 != -1 else -1,
        "Misc3": idx_misc3 if idx_misc3 != -1 else -1,
        "Misc4": idx_misc4 if idx_misc4 != -1 else -1,
        "TxRxHoriSep": idx_txrx_h if idx_txrx_h != -1 else -1,
        "TxRxVertSep": idx_txrx_v if idx_txrx_v != -1 else -1,
    }

    layout = {
        "ch1_label": ch1_label,
        "ch2_label": ch2_label,
        "gates_ch1": gates_ch1,
        "gates_ch2": gates_ch2,
        "std_ch1": std_ch1,
        "std_ch2": std_ch2,
        "current_ch1": current_ch1 if current_ch1 > 0 else -1,
        "current_ch2": current_ch2 if current_ch2 > 0 else -1,
        "field_indices": field_indices,
    }

    return "\n".join(lines_out), info, layout


# ---------- 3-row mapping view helpers ----------

def build_core_mapping(layout, pos, first_values, overrides):
    """
    Build a table-like structure for core fields including Misc1–Misc4, TxRxHoriSep/Vert, and currents.
    Returns list of dicts with:
    - Field
    - ALC index
    - XYZ index
    - XYZ header
    - First value
    - Status
    - Override_header (initial from overrides if present)
    """
    rows = []

    fi = layout["field_indices"]

    # Field name, default "expected" header name (only used if no override & present)
    core_fields = [
        ("Line", "Line"),
        ("Date", "Date"),
        ("Time", "Time"),
        ("Topography", "DEM"),
        ("TxAltitude", "Height"),
        ("TxPitch", "AngleX"),
        ("TxRoll", "AngleY"),
        ("UTMX", "E"),
        ("UTMY", "N"),
        ("Magnetic", "TMI"),
        ("PowerLineMonitor", "PLNI"),
        ("Misc1", None),
        ("Misc2", None),
        ("Misc3", None),
        ("Misc4", None),
        ("TxRxHoriSep", None),
        ("TxRxVertSep", None),
    ]

    def find_name_by_index(idx):
        for name, i in pos.items():
            if i == idx:
                return name
        return ""

    # Core fields
    for field_name, default_header in core_fields:
        alc_idx = fi.get(field_name, -1)
        if alc_idx > 0:
            xyz_idx = alc_idx
            xyz_header = find_name_by_index(xyz_idx)
            first_val = first_values[xyz_idx - 1] if len(first_values) >= xyz_idx else ""
            status = "✅ OK"
        else:
            xyz_idx = -1
            xyz_header = ""
            first_val = ""
            status = "❌ MISSING"

        rows.append(
            {
                "Field": field_name,
                "ALC index": alc_idx,
                "XYZ index": xyz_idx if xyz_idx > 0 else "",
                "XYZ header": xyz_header if xyz_header else "Not found",
                "First value": first_val,
                "Status": status,
                "Override_header": overrides.get(field_name, ""),
            }
        )

    # Currents
    def add_current(field_name, alc_idx):
        if alc_idx == 0:
            # 0 means "not applicable" (e.g. no Ch02 at all), skip
            return
        if alc_idx > 0:
            xyz_idx = alc_idx
            xyz_header = find_name_by_index(xyz_idx)
            first_val = first_values[xyz_idx - 1] if len(first_values) >= xyz_idx else ""
            status = "✅ OK"
        else:
            xyz_idx = -1
            xyz_header = ""
            first_val = ""
            status = "❌ MISSING"

        rows.append(
            {
                "Field": field_name,
                "ALC index": alc_idx,
                "XYZ index": xyz_idx if xyz_idx > 0 else "",
                "XYZ header": xyz_header if xyz_header else "Not found",
                "First value": first_val,
                "Status": status,
                "Override_header": overrides.get(field_name, ""),
            }
        )

    add_current("Current_Ch01", layout["current_ch1"])
    if layout["current_ch2"] != 0:
        add_current("Current_Ch02", layout["current_ch2"])

    return rows


def build_gate_mapping(layout, pos, first_values, channel=1, max_rows=10):
    rows = []
    if channel == 1:
        gates = layout["gates_ch1"]
        tag = "Ch01"
    else:
        gates = layout["gates_ch2"]
        tag = "Ch02"

    for i, name in enumerate(gates, start=1):
        if i > max_rows:
            break
        idx = pos[name]
        first_val = first_values[idx - 1] if len(first_values) >= idx else ""
        rows.append(
            {
                "ALC entry": f"Gate_{tag}_{i:02d} = {idx}",
                "XYZ column": f"{idx} → {name}",
                "First value": first_val,
                "Status": "✅ OK",
            }
        )
    return rows


def build_std_mapping(layout, pos, first_values, channel=1, max_rows=10):
    rows = []
    if channel == 1:
        stds = layout["std_ch1"]
        tag = "Ch01"
    else:
        stds = layout["std_ch2"]
        tag = "Ch02"

    for i, name in enumerate(stds, start=1):
        if i > max_rows:
            break
        idx = pos[name]
        first_val = first_values[idx - 1] if len(first_values) >= idx else ""
        rows.append(
            {
                "ALC entry": f"STD_{tag}_{i:02d} = {idx}",
                "XYZ column": f"{idx} → {name}",
                "First value": first_val,
                "Status": "✅ OK",
            }
        )
    return rows


# ---------- Streamlit app ----------

st.title("XYZ → SkyTEM .ALC builder")

st.write(
    "Upload a **SkyTEM XYZ** file. "
    "The app will read the header, detect LM/HM gates and relative uncertainties "
    "(supports both `RelUnc_LM_Z_dBdt...` and `RelUnc_SWch1/2_G01...` styles), "
    "generate an `.ALC` format file, and show a 3-row mapping view with **Status**.\n\n"
    "- Core mapping includes the main ALC parameters: Line, Date, Time, Topography, TxAltitude, "
    "TxPitch, TxRoll, UTMX, UTMY, Magnetic, PowerLineMonitor, Misc1–Misc4, TxRxHoriSep, TxRxVertSep, "
    "Current_Ch01, Current_Ch02.\n"
    "- Any ALC parameter that cannot be identified from the XYZ header is marked as **❌ MISSING**.\n"
    "- For missing fields you can pick an **Override header** directly in the table."
)

uploaded = st.file_uploader("Upload XYZ file", type=["xyz", "txt", "dat", "csv"])

system_name = st.text_input("System name in ALC", value="SkyTEM XYZ")

ch1_label = st.selectbox("Channel 1 type (Ch01)", ["LM", "HM"], index=0)
ch2_selection = st.selectbox("Channel 2 type (Ch02, optional)", ["None", "LM", "HM"], index=2)
if ch2_selection == "None":
    ch2_label = None
else:
    ch2_label = ch2_selection

max_rows_to_show = st.slider("Number of gates/STD entries to show in mapping", 5, 40, 10)

if "core_overrides" not in st.session_state:
    st.session_state["core_overrides"] = {}

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8", errors="ignore")
        all_lines = [ln for ln in content.splitlines() if ln.strip()]

        if len(all_lines) < 2:
            st.error("File must contain at least a header line and one data line.")
        else:
            header_line = all_lines[0]
            first_data_line = all_lines[1]

            st.subheader("Detected header line")
            st.code(header_line, language="text")

            cols, pos = parse_header_line(header_line)
            first_values = parse_first_data_line(first_data_line)

            st.subheader("Detected columns (first 60)")
            st.write(cols[:60])

            overrides = st.session_state["core_overrides"]

            # Build ALC using current overrides
            alc_text, info, layout = build_alc_text(
                cols,
                pos,
                system_name=system_name,
                ch1_label=ch1_label,
                ch2_label=ch2_label,
                overrides=overrides,
            )

            st.subheader("Channel summary")
            st.json(info)

            # ----- Core mapping with dynamic override cells -----
            st.subheader("Core field mapping (with dynamic Override header)")

            core_rows = build_core_mapping(layout, pos, first_values, overrides)
            df_core = pd.DataFrame(core_rows)

            edited_df = st.data_editor(
                df_core,
                column_config={
                    "Override_header": st.column_config.SelectboxColumn(
                        "Override header (optional)",
                        options=[""] + list(pos.keys()),
                        help=(
                            "Pick a header column to map this ALC field to, "
                            "if missing or to override the default mapping."
                        ),
                    )
                },
                disabled=["Field", "ALC index", "XYZ index", "XYZ header", "First value", "Status"],
                use_container_width=True,
                key="core_editor",
            )

            # Update overrides from edited table
            new_overrides = {}
            for _, row in edited_df.iterrows():
                field = row["Field"]
                ov = row["Override_header"]
                if isinstance(ov, str) and ov:
                    new_overrides[field] = ov

            st.session_state["core_overrides"] = new_overrides

            # ----- Gate / STD mapping -----
            st.subheader(f"Gate mapping (Ch01 – {layout['ch1_label']})")
            gate_ch1_rows = build_gate_mapping(layout, pos, first_values, channel=1, max_rows=max_rows_to_show)
            if gate_ch1_rows:
                st.table(gate_ch1_rows)
            else:
                st.info("No gates found for Channel 1 in this file.")

            if info["channels_number"] == 2:
                st.subheader(f"Gate mapping (Ch02 – {layout['ch2_label']})")
                gate_ch2_rows = build_gate_mapping(layout, pos, first_values, channel=2, max_rows=max_rows_to_show)
                if gate_ch2_rows:
                    st.table(gate_ch2_rows)
                else:
                    st.info("No gates found for Channel 2 in this file.")

            st.subheader(f"STD mapping (Ch01 – {layout['ch1_label']})")
            std_ch1_rows = build_std_mapping(layout, pos, first_values, channel=1, max_rows=max_rows_to_show)
            if std_ch1_rows:
                st.table(std_ch1_rows)
            else:
                st.info("No STD / uncertainty columns found for Channel 1.")

            if info["channels_number"] == 2:
                st.subheader(f"STD mapping (Ch02 – {layout['ch2_label']})")
                std_ch2_rows = build_std_mapping(layout, pos, first_values, channel=2, max_rows=max_rows_to_show)
                if std_ch2_rows:
                    st.table(std_ch2_rows)
                else:
                    st.info("No STD / uncertainty columns found for Channel 2.")

            # ----- ALC preview + download -----
            st.subheader("Preview of generated .ALC")
            preview_lines = alc_text.splitlines()
            if len(preview_lines) > 150:
                preview_show = "\n".join(preview_lines[:150]) + "\n..."
            else:
                preview_show = alc_text
            st.code(preview_show, language="text")

            st.download_button(
                "Download .ALC",
                data=alc_text,
                file_name="output.ALC",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Error while processing file: {e}")

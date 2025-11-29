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
    """Parse the first data line into a list of values, split on whitespace."""
    return raw.strip().split()


def find_indexed_columns_with_prefix(cols, prefix):
    """
    Find columns like prefix[0], prefix[1], ...
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
    """Try a list of prefixes in order, return the first non-empty list."""
    for p in prefixes:
        lst = find_indexed_columns_with_prefix(cols, p)
        if lst:
            return lst
    return []


def find_current_index_auto(pos, label):
    """
    Find auto-detected current column index for LM or HM, case-insensitive.
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
    index_overrides=None,
):
    """
    Build ALC text given parsed header and mapping.
    - ch1_label / ch2_label: 'LM' or 'HM' (channel roles)
    - index_overrides: dict[field_name -> index] that overrides auto-detected indices
    Returns: (alc_text, info, layout)
    """
    index_overrides = index_overrides or {}

    def apply_index(field_name, auto_index):
        """Apply index override for a field if present, otherwise use auto_index."""
        if field_name in index_overrides:
            try:
                v = int(index_overrides[field_name])
            except (TypeError, ValueError):
                v = -1
            return v
        return auto_index

    # -------- Gate columns --------
    lm_gates = get_first_non_empty_indexed(cols, ["LM_Z_dBdt"])
    hm_gates = get_first_non_empty_indexed(cols, ["HM_Z_dBdt", "HM_Z_dbdt"])

    channels_present = []
    if lm_gates:
        channels_present.append("LM")
    if hm_gates:
        channels_present.append("HM")

    if not channels_present:
        raise ValueError("No LM/HM gate columns detected in header.")

    # Resolve channel roles (Ch01 / Ch02) with fallback, supports HM-only / LM-only.
    if ch1_label not in channels_present:
        ch1_label = channels_present[0]
    if ch2_label not in channels_present or ch2_label == ch1_label:
        ch2_label = None

    channels_number = 1 if ch2_label is None else 2

    # -------- STD columns (relative uncertainties) --------
    # Support multiple name patterns:
    # - RelUnc_LM_Z_dBdt_Merge / RelUnc_HM_Z_dBdt
    # - RelUnc_SWch1_G01 / RelUnc_SWch2_G01
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

    if ch1_label == "LM":
        ch1_current_auto = find_current_index_auto(pos, "LM")
    else:
        ch1_current_auto = find_current_index_auto(pos, "HM")
    current_ch1 = apply_index("Current_Ch01", ch1_current_auto)

    if ch2_label:
        gates_ch2 = gates_for_label(ch2_label)
        std_ch2 = std_for_label(ch2_label)
        if ch2_label == "LM":
            ch2_current_auto = find_current_index_auto(pos, "LM")
        else:
            ch2_current_auto = find_current_index_auto(pos, "HM")
        current_ch2 = apply_index("Current_Ch02", ch2_current_auto)
    else:
        gates_ch2, std_ch2 = [], []
        current_ch2 = 0  # 0 = no Ch02 at all

    # ----- Auto indices for core fields -----
    def auto_idx_from_header(name):
        return pos.get(name, -1)

    idx_date_auto = auto_idx_from_header("Date")
    idx_line_auto = auto_idx_from_header("Line")
    idx_time_auto = auto_idx_from_header("Time")
    idx_anglex_auto = auto_idx_from_header("AngleX")
    idx_angley_auto = auto_idx_from_header("AngleY")
    idx_height_auto = auto_idx_from_header("Height")
    idx_e_auto = auto_idx_from_header("E")
    idx_n_auto = auto_idx_from_header("N")
    idx_dem_auto = auto_idx_from_header("DEM")
    idx_tmi_auto = auto_idx_from_header("TMI")
    idx_plni_auto = auto_idx_from_header("PLNI")

    # Misc, TxRx: default is unassigned (-1) unless user sets override
    idx_misc1_auto = -1
    idx_misc2_auto = -1
    idx_misc3_auto = -1
    idx_misc4_auto = -1
    idx_txrx_h_auto = -1
    idx_txrx_v_auto = -1

    # Apply overrides
    idx_date = apply_index("Date", idx_date_auto)
    idx_line = apply_index("Line", idx_line_auto)
    idx_time = apply_index("Time", idx_time_auto)
    idx_anglex = apply_index("TxPitch", idx_anglex_auto)
    idx_angley = apply_index("TxRoll", idx_angley_auto)
    idx_height = apply_index("TxAltitude", idx_height_auto)
    idx_e = apply_index("UTMX", idx_e_auto)
    idx_n = apply_index("UTMY", idx_n_auto)
    idx_dem = apply_index("Topography", idx_dem_auto)
    idx_tmi = apply_index("Magnetic", idx_tmi_auto)
    idx_plni = apply_index("PowerLineMonitor", idx_plni_auto)

    idx_misc1 = apply_index("Misc1", idx_misc1_auto)
    idx_misc2 = apply_index("Misc2", idx_misc2_auto)
    idx_misc3 = apply_index("Misc3", idx_misc3_auto)
    idx_misc4 = apply_index("Misc4", idx_misc4_auto)

    idx_txrx_h = apply_index("TxRxHoriSep", idx_txrx_h_auto)
    idx_txrx_v = apply_index("TxRxVertSep", idx_txrx_v_auto)

    # ----- Build ALC text -----
    def kv(key, val):
        return f"{key:<22}= {val}"

    lines_out = []
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
    lines_out.append("")

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


# ---------- Mapping view helpers ----------

def build_core_mapping(layout, pos, first_values, index_overrides, n_cols):
    """
    Build table rows for core fields including Misc1–Misc4, TxRxHoriSep/Vert, and currents.
    Each row: Field, Index, XYZ header, First value, Status
    """
    rows = []

    fi = layout["field_indices"]

    core_fields = [
        "Line",
        "Date",
        "Time",
        "Topography",
        "TxAltitude",
        "TxPitch",
        "TxRoll",
        "UTMX",
        "UTMY",
        "Magnetic",
        "PowerLineMonitor",
        "Misc1",
        "Misc2",
        "Misc3",
        "Misc4",
        "TxRxHoriSep",
        "TxRxVertSep",
    ]

    def find_name_by_index(idx):
        for name, i in pos.items():
            if i == idx:
                return name
        return ""

    def status_for_index(idx):
        if idx <= 0 or idx > n_cols:
            return "❌ MISSING"
        return "✅ OK"

    def first_val_for_index(idx):
        if idx <= 0 or idx > len(first_values):
            return ""
        return first_values[idx - 1]

    # Core fields
    for field_name in core_fields:
        # Index currently used in ALC
        base_idx = fi.get(field_name, -1)
        # If user has override, show that in the table
        idx = index_overrides.get(field_name, base_idx)
        if idx is None:
            idx = -1
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = -1

        xyz_header = find_name_by_index(idx) if 0 < idx <= n_cols else ""
        status = status_for_index(idx)
        first_val = first_val_for_index(idx)

        rows.append(
            {
                "Field": field_name,
                "Index": idx,
                "XYZ header": xyz_header if xyz_header else "Not found",
                "First value": first_val,
                "Status": status,
            }
        )

    # Currents
    def add_current_row(field_name, alc_idx):
        if alc_idx == 0:
            # no Ch02 at all, skip
            return
        idx = index_overrides.get(field_name, alc_idx)
        if idx is None:
            idx = -1
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = -1

        xyz_header = find_name_by_index(idx) if 0 < idx <= n_cols else ""
        status = status_for_index(idx)
        first_val = first_val_for_index(idx)

        rows.append(
            {
                "Field": field_name,
                "Index": idx,
                "XYZ header": xyz_header if xyz_header else "Not found",
                "First value": first_val,
                "Status": status,
            }
        )

    add_current_row("Current_Ch01", layout["current_ch1"])
    if layout["current_ch2"] != 0:
        add_current_row("Current_Ch02", layout["current_ch2"])

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
    "Upload a **SkyTEM XYZ** file. The app:\n"
    "- Detects LM/HM gates + relative uncertainties (both old and new naming styles)\n"
    "- Generates a `.ALC` format file\n"
    "- Shows a core mapping table where **ALC index = XYZ index**.\n\n"
    "You can change the index using the little up/down arrows (scroll wheel) per row; "
    "the header name and first value will update automatically on the next run."
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

if "index_overrides" not in st.session_state:
    st.session_state["index_overrides"] = {}

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
            n_cols = len(cols)

            st.subheader("Detected columns (first 60)")
            st.write(cols[:60])

            index_overrides = st.session_state["index_overrides"]

            # Build ALC using current index overrides
            alc_text, info, layout = build_alc_text(
                cols,
                pos,
                system_name=system_name,
                ch1_label=ch1_label,
                ch2_label=ch2_label,
                index_overrides=index_overrides,
            )

            st.subheader("Channel summary")
            st.json(info)

            # ----- Core mapping with editable Index -----
            st.subheader("Core field mapping (edit Index to remap)")

            core_rows = build_core_mapping(layout, pos, first_values, index_overrides, n_cols)
            df_core = pd.DataFrame(core_rows)

            edited_df = st.data_editor(
                df_core,
                column_config={
                    "Index": st.column_config.NumberColumn(
                        "Index",
                        min_value=-1,
                        max_value=n_cols,
                        step=1,
                        help="ALC index = XYZ index. Set -1 for unused.",
                    )
                },
                disabled=["Field", "XYZ header", "First value", "Status"],
                use_container_width=True,
                key="core_editor",
            )

            # Update index_overrides from edited table
            new_index_overrides = {}
            for _, row in edited_df.iterrows():
                field = row["Field"]
                idx = row["Index"]
                if pd.isna(idx):
                    continue
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    idx_int = -1
                new_index_overrides[field] = idx_int

            st.session_state["index_overrides"] = new_index_overrides

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

import streamlit as st
import pandas as pd


# ---------- Parsing helpers ----------

def parse_header_line(raw: str):
    line = raw.strip()
    if line.startswith("/"):
        line = line[1:]
    cols = line.split()
    pos = {name: i + 1 for i, name in enumerate(cols)}  # 1-based
    return cols, pos


def parse_first_data_line(raw: str):
    return raw.strip().split()


def find_indexed_columns_with_prefix(cols, prefix):
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
    for p in prefixes:
        lst = find_indexed_columns_with_prefix(cols, p)
        if lst:
            return lst
    return []


def find_current_index_auto(pos, label):
    if not label:
        return 0
    if label == "LM":
        candidates = ["LMcurrent", "LMCurrent"]
    else:
        candidates = ["HMcurrent", "HMCurrent"]

    for cand in candidates:
        for name, idx in pos.items():
            if name.lower() == cand.lower():
                return idx
    return -1


# ---------- ALC builder ----------

CORE_FIELDS = [
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
    # "TxRxHoriSep",
    # "TxRxVertSep",
]


def build_alc_text(
    cols,
    pos,
    system_name,
    ch1_label,
    ch2_label,
    field_indices,
    current_ch1_idx,
    current_ch2_idx,
):
    """Build ALC text from explicit indices."""
    # Gates
    lm_gates = get_first_non_empty_indexed(cols, ["LM_Z_dBdt"])
    hm_gates = get_first_non_empty_indexed(cols, ["HM_Z_dBdt", "HM_Z_dbdt"])

    channels_present = []
    if lm_gates:
        channels_present.append("LM")
    if hm_gates:
        channels_present.append("HM")

    if not channels_present:
        raise ValueError("No LM/HM gate columns detected in header.")

    # Resolve channel roles
    if ch1_label not in channels_present:
        ch1_label = channels_present[0]
    if ch2_label not in channels_present or ch2_label == ch1_label:
        ch2_label = None

    channels_number = 1 if ch2_label is None else 2

    def std_for_label(label):
        if label == "LM":
            prefixes = [
                "RelUnc_LM_Z_dBdt",
                "RelUnc_LM",
                "RelUnc_SWch1_G01",
                "RelUnc_SWch1",
            ]
        else:
            prefixes = [
                "RelUnc_HM_Z_dBdt",
                "RelUnc_HM",
                "RelUnc_SWch2_G01",
                "RelUnc_SWch2",
            ]
        return get_first_non_empty_indexed(cols, prefixes)

    def gates_for_label(label):
        return lm_gates if label == "LM" else hm_gates

    gates_ch1 = gates_for_label(ch1_label)
    std_ch1 = std_for_label(ch1_label)

    if ch2_label:
        gates_ch2 = gates_for_label(ch2_label)
        std_ch2 = std_for_label(ch2_label)
    else:
        gates_ch2, std_ch2 = [], []

    # Extract core indices
    idx_line = field_indices.get("Line", -1)
    idx_date = field_indices.get("Date", -1)
    idx_time = field_indices.get("Time", -1)
    idx_dem = field_indices.get("Topography", -1)
    idx_height = field_indices.get("TxAltitude", -1)
    idx_txpitch = field_indices.get("TxPitch", -1)
    idx_txroll = field_indices.get("TxRoll", -1)
    idx_e = field_indices.get("UTMX", -1)
    idx_n = field_indices.get("UTMY", -1)
    idx_mag = field_indices.get("Magnetic", -1)
    idx_plni = field_indices.get("PowerLineMonitor", -1)
    idx_misc1 = field_indices.get("Misc1", -1)
    idx_misc2 = field_indices.get("Misc2", -1)
    idx_misc3 = field_indices.get("Misc3", -1)
    idx_misc4 = field_indices.get("Misc4", -1)
    # idx_txrx_h = field_indices.get("TxRxHoriSep", -1)
    # idx_txrx_v = field_indices.get("TxRxVertSep", -1)

    def kv(key, val):
        return f"{key:<22}= {val}"

    lines_out = []
    lines_out.append(kv("Version", 2))
    lines_out.append(kv("System", system_name))
    lines_out.append(kv("ChannelsNumber", channels_number))
    lines_out.append(kv("Date", idx_date))
    lines_out.append(kv("Dummy", "*"))
    lines_out.append(kv("Line", idx_line))
    lines_out.append(kv("Magnetic", idx_mag))
    lines_out.append(kv("Misc1", idx_misc1))
    lines_out.append(kv("Misc2", idx_misc2))
    lines_out.append(kv("Misc3", idx_misc3))
    lines_out.append(kv("Misc4", idx_misc4))
    # lines_out.append(kv("RxPitch", -1))
    # lines_out.append(kv("RxRoll", -1))
    lines_out.append(kv("Time", idx_time))
    lines_out.append(kv("Topography", idx_dem))
    lines_out.append(kv("TxAltitude", idx_height))
    # lines_out.append(kv("TxOffTime", -1))
    # lines_out.append(kv("TxOnTime", -1))
    # lines_out.append(kv("TxPeakTime", -1))
    lines_out.append(kv("TxPitch", idx_txpitch))
    lines_out.append(kv("TxRoll", idx_txroll))
    # lines_out.append(kv("TxRxHoriSep", idx_txrx_h))
    # lines_out.append(kv("TxRxVertSep", idx_txrx_v))
    lines_out.append(kv("UTMX", idx_e))
    lines_out.append(kv("UTMY", idx_n))
    lines_out.append(kv("Current_Ch01", current_ch1_idx))
    if channels_number == 2:
        lines_out.append(kv("Current_Ch02", current_ch2_idx))
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

    # InUse all -1
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
        "field_indices": field_indices,
        "current_ch1": current_ch1_idx,
        "current_ch2": current_ch2_idx,
        "gates_ch1": gates_ch1,
        "gates_ch2": gates_ch2,
        "std_ch1": std_ch1,
        "std_ch2": std_ch2,
    }

    return "\n".join(lines_out), info, layout


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
        val = first_values[idx - 1] if len(first_values) >= idx else ""
        rows.append(
            {
                "ALC entry": f"Gate_{tag}_{i:02d} = {idx}",
                "XYZ column": f"{idx} → {name}",
                "First value": val,
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
        val = first_values[idx - 1] if len(first_values) >= idx else ""
        rows.append(
            {
                "ALC entry": f"STD_{tag}_{i:02d} = {idx}",
                "XYZ column": f"{idx} → {name}",
                "First value": val,
                "Status": "✅ OK",
            }
        )
    return rows


# ---------- Streamlit app ----------

st.title("XYZ → SkyTEM .ALC builder (row-by-row spinners)")

st.write(
    "Upload a SkyTEM XYZ file. This app:\n"
    "- Auto-detects sensible default indices for core ALC fields\n"
    "- Lets you override each **index** with a spinner\n"
    "- Shows header name, first value, and status on \n"
    "- Generates an `.ALC` file where **ALC index = XYZ index** (or -1 for unused)."
)

uploaded = st.file_uploader("Upload XYZ file", type=["xyz", "txt", "dat", "csv"])

system_name = st.text_input("System name in ALC", value="SkyTEM XYZ")

ch1_label = st.selectbox("Channel 1 type (Ch01)", ["LM", "HM"], index=0)
ch2_sel = st.selectbox("Channel 2 type (Ch02, optional)", ["None", "LM", "HM"], index=2)
ch2_label = None if ch2_sel == "None" else ch2_sel

max_rows_to_show = st.slider("Number of gates/STD entries to show", 5,140, 10)

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in content.splitlines() if ln.strip()]

        if len(lines) < 2:
            st.error("File must contain at least a header line and one data line.")
        else:
            header_line = lines[0]
            first_data_line = lines[1]

            st.subheader("Detected header line")
            st.code(header_line, language="text")

            cols, pos = parse_header_line(header_line)
            first_values = parse_first_data_line(first_data_line)
            n_cols = len(cols)

            st.subheader("Detected columns")
            st.write(cols)

            # --- Auto core indices from header names ---
            def auto_idx(name):
                return pos.get(name, -1)

            auto_field_indices = {
                "Line": auto_idx("Line"),
                "Date": auto_idx("Date"),
                "Time": auto_idx("Time"),
                "Topography": auto_idx("DEM"),
                "TxAltitude": auto_idx("Height"),
                "TxPitch": auto_idx("AngleX"),
                "TxRoll": auto_idx("AngleY"),
                "UTMX": auto_idx("E"),
                "UTMY": auto_idx("N"),
                "Magnetic": auto_idx("TMI"),
                "PowerLineMonitor": auto_idx("PLNI"),
                "Misc1": -1,
                "Misc2": -1,
                "Misc3": -1,
                "Misc4": -1,
                # "TxRxHoriSep": -1,
                # "TxRxVertSep": -1,
            }

            auto_current_ch1 = find_current_index_auto(pos, ch1_label)
            auto_current_ch2 = find_current_index_auto(pos, ch2_label) if ch2_label else 0

            # Helpers for preview
            def find_name_by_index(idx):
                for name, i in pos.items():
                    if i == idx:
                        return name
                return ""

            def first_val(idx):
                if 1 <= idx <= len(first_values):
                    return first_values[idx - 1]
                return ""

            def status(idx):
                if 1 <= idx <= n_cols:
                    return "✅ OK"
                return "❌ MISSING"

            st.subheader("Core field mapping (spinner + preview per row)")

            # Header row (visual only)
            h1, h2, h3, h4, h5 = st.columns([1.3, 1, 2, 2, 1])
            with h1:
                st.markdown("**Field**")
            with h2:
                st.markdown("**Index**")
            with h3:
                st.markdown("**XYZ header**")
            with h4:
                st.markdown("**First value**")
            with h5:
                st.markdown("**Status**")

            field_indices = {}

            # Build rows with spinners + preview
            for field in CORE_FIELDS:
                default_idx = auto_field_indices[field]

                c1, c2, c3, c4, c5 = st.columns([1.3, 1, 2, 2, 1])

                with c1:
                    st.write(field)

                with c2:
                    idx_val = st.number_input(
                        label=f"Index for {field}",
                        min_value=-1,
                        max_value=n_cols,
                        value=int(default_idx),
                        step=1,
                        key=f"idx_{field}",
                        label_visibility="collapsed",
                    )

                idx_int = int(idx_val)
                field_indices[field] = idx_int

                header_name = find_name_by_index(idx_int) if idx_int > 0 else "Not found"
                fv = first_val(idx_int)
                st_status = status(idx_int)

                with c3:
                    st.write(header_name)
                with c4:
                    st.write(fv)
                with c5:
                    st.write(st_status)

            # Currents rows
            st.markdown("---")
            st.markdown("**Current channels**")

            c1, c2, c3, c4, c5 = st.columns([1.3, 1, 2, 2, 1])
            with c1:
                st.write("Current_Ch01")
            with c2:
                current_ch1_idx = st.number_input(
                    label="Index for Current_Ch01",
                    min_value=-1,
                    max_value=n_cols,
                    value=int(auto_current_ch1),
                    step=1,
                    key="idx_Current_Ch01",
                    label_visibility="collapsed",
                )
            current_ch1_idx = int(current_ch1_idx)
            with c3:
                st.write(find_name_by_index(current_ch1_idx) if current_ch1_idx > 0 else "Not found")
            with c4:
                st.write(first_val(current_ch1_idx))
            with c5:
                st.write(status(current_ch1_idx))

            current_ch2_idx = 0
            if ch2_label is not None:
                c1, c2, c3, c4, c5 = st.columns([1.3, 1, 2, 2, 1])
                with c1:
                    st.write("Current_Ch02")
                with c2:
                    current_ch2_idx = st.number_input(
                        label="Index for Current_Ch02",
                        min_value=-1,
                        max_value=n_cols,
                        value=int(auto_current_ch2),
                        step=1,
                        key="idx_Current_Ch02",
                        label_visibility="collapsed",
                    )
                current_ch2_idx = int(current_ch2_idx)
                with c3:
                    st.write(find_name_by_index(current_ch2_idx) if current_ch2_idx > 0 else "Not found")
                with c4:
                    st.write(first_val(current_ch2_idx))
                with c5:
                    st.write(status(current_ch2_idx))

            # --- Build ALC with these indices ---
            alc_text, info, layout = build_alc_text(
                cols,
                pos,
                system_name,
                ch1_label,
                ch2_label,
                field_indices,
                current_ch1_idx,
                current_ch2_idx,
            )

            st.subheader("Channel summary")
            st.json(info)

            # --- Gate / STD mapping ---
            st.subheader(f"Gate mapping (Ch01 – {ch1_label})")
            gate_ch1_rows = build_gate_mapping(
                layout, pos, first_values, channel=1, max_rows=max_rows_to_show
            )
            if gate_ch1_rows:
                st.table(pd.DataFrame(gate_ch1_rows))
            else:
                st.info("No gates found for Channel 1.")

            if info["channels_number"] == 2:
                st.subheader(f"Gate mapping (Ch02 – {ch2_label})")
                gate_ch2_rows = build_gate_mapping(
                    layout, pos, first_values, channel=2, max_rows=max_rows_to_show
                )
                if gate_ch2_rows:
                    st.table(pd.DataFrame(gate_ch2_rows))
                else:
                    st.info("No gates found for Channel 2.")

            st.subheader(f"STD mapping (Ch01 – {ch1_label})")
            std_ch1_rows = build_std_mapping(
                layout, pos, first_values, channel=1, max_rows=max_rows_to_show
            )
            if std_ch1_rows:
                st.table(pd.DataFrame(std_ch1_rows))
            else:
                st.info("No STD columns for Channel 1.")

            if info["channels_number"] == 2:
                st.subheader(f"STD mapping (Ch02 – {ch2_label})")
                std_ch2_rows = build_std_mapping(
                    layout, pos, first_values, channel=2, max_rows=max_rows_to_show
                )
                if std_ch2_rows:
                    st.table(pd.DataFrame(std_ch2_rows))
                else:
                    st.info("No STD columns for Channel 2.")

            # --- ALC preview + download ---
            st.subheader("Preview of generated .ALC")
            lines_out = alc_text.splitlines()
            if len(lines_out) > 350:
                st.code("\n".join(lines_out[:350]) + "\n...", language="text")
            else:
                st.code(alc_text, language="text")

            st.download_button(
                "Download .ALC",
                data=alc_text,
                file_name="output.ALC",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Error while processing file: {e}")

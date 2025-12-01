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


def find_indexed_columns_with_filter(cols, prefixes, must_contain=None, must_not_contain=None):
    """
    Generic helper:
    - prefixes: list of starting patterns (startswith)
    - must_contain: list of substrings that MUST appear (if not None)
    - must_not_contain: list of substrings that MUST NOT appear (if not None)
    Returns list of column names sorted by the index inside [idx].
    """
    must_contain = must_contain or []
    must_not_contain = must_not_contain or []

    found = []
    for name in cols:
        if not any(name.startswith(p) for p in prefixes):
            continue
        if must_contain and not any(s in name for s in must_contain):
            continue
        if must_not_contain and any(s in name for s in must_not_contain):
            continue

        if "[" in name and "]" in name:
            try:
                idx_str = name.split("[", 1)[1].split("]", 1)[0]
                gate_index = int(idx_str)
                found.append((gate_index, name))
            except ValueError:
                continue

    found_sorted = [n for _, n in sorted(found)]
    return found_sorted


def get_first_non_empty_indexed(cols, prefixes, must_contain=None, must_not_contain=None):
    """Try a list of prefixes in order, return the first non-empty list."""
    for p in prefixes:
        lst = find_indexed_columns_with_filter(
            cols, [p], must_contain=must_contain, must_not_contain=must_not_contain
        )
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


def slice_by_range(lst, first_idx, last_idx):
    """Slice a list by 1-based [first_idx, last_idx] inclusive."""
    if not lst:
        return []
    n = len(lst)
    if first_idx < 1:
        first_idx = 1
    if last_idx < first_idx:
        last_idx = first_idx
    if first_idx > n:
        return []
    if last_idx > n:
        last_idx = n
    return lst[first_idx - 1:last_idx]


# ---------- Channel detection ----------

def detect_channel_layout(cols, pos, ch1_label, ch2_label):
    """
    Detect LM/HM gates, STD, InUse and resolve which label is Ch01 / Ch02.

    Returns dict:
      {
        'channels_number': int,
        'ch1_label': 'LM' or 'HM',
        'ch2_label': 'LM'/'HM'/None,
        'gates_ch1_all': [...],
        'gates_ch2_all': [...],
        'std_ch1_all': [...],
        'std_ch2_all': [...],
        'inuse_ch1_all': [...],
        'inuse_ch2_all': [...],
      }
    """

    # ---- Gates (exclude STD/InUse by name) ----
    lm_gates_all = get_first_non_empty_indexed(
        cols,
        prefixes=["LM_Z_dBdt"],
        must_not_contain=["RelUnc", "InUse"],
    )

    hm_gates_all = get_first_non_empty_indexed(
        cols,
        prefixes=[
            "HM_Z_dBdt_XYcorr_Norm_merged",
            "HM_Z_dBdt",
            "HM_Z_dbdt",
        ],
        must_not_contain=["RelUnc", "InUse"],
    )

    channels_present = []
    if lm_gates_all:
        channels_present.append("LM")
    if hm_gates_all:
        channels_present.append("HM")

    if not channels_present:
        raise ValueError("No LM/HM gate columns detected in header.")

    # Resolve channel roles (Ch01 / Ch02)
    if ch1_label not in channels_present:
        ch1_label = channels_present[0]

    if ch2_label not in channels_present or ch2_label == ch1_label:
        ch2_label = None

    channels_number = 1 if ch2_label is None else 2

    # ---- STD detection ----
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
        return get_first_non_empty_indexed(
            cols, prefixes=prefixes, must_contain=["RelUnc"]
        )

    # ---- Gates by label ----
    def gates_for_label(label):
        return lm_gates_all if label == "LM" else hm_gates_all

    # ---- InUse detection per label ----
    def inuse_for_label(label):
        if label == "LM":
            prefixes = [
                "LM_Z_dBdt_InUse_merged",
                "LM_Z_dBdt_InUse",
                "InUse_LM_Z_dBdt",
            ]
        else:  # HM
            prefixes = [
                "HM_Z_dBdt_InUse_merged",
                "HM_Z_dBdt_InUse",
                "InUse_HM_Z_dBdt",
            ]
        return get_first_non_empty_indexed(
            cols, prefixes=prefixes, must_contain=["InUse"]
        )

    # Channel 1
    gates_ch1_all = gates_for_label(ch1_label)
    std_ch1_all = std_for_label(ch1_label)
    inuse_ch1_all = inuse_for_label(ch1_label)

    # Channel 2
    if ch2_label:
        gates_ch2_all = gates_for_label(ch2_label)
        std_ch2_all = std_for_label(ch2_label)
        inuse_ch2_all = inuse_for_label(ch2_label)
    else:
        gates_ch2_all, std_ch2_all, inuse_ch2_all = [], [], []

    return {
        "channels_number": channels_number,
        "ch1_label": ch1_label,
        "ch2_label": ch2_label,
        "gates_ch1_all": gates_ch1_all,
        "gates_ch2_all": gates_ch2_all,
        "std_ch1_all": std_ch1_all,
        "std_ch2_all": std_ch2_all,
        "inuse_ch1_all": inuse_ch1_all,
        "inuse_ch2_all": inuse_ch2_all,
    }


# ---------- ALC builder ----------

CORE_FIELDS = [
    "Line",
    "Date",
    "Time",
    "Topography",
    "TxAltitude",
    "UTMX",
    "UTMY",
    "Magnetic",
    "PowerLineMonitor",
    "Misc1",
    "Misc2",
    "Misc3",
    "Misc4",
]


def build_alc_text(
    system_name,
    field_indices,
    current_ch1_idx,
    current_ch2_idx,
    channels_number,
    ch1_label,
    ch2_label,
    gates_ch1,
    gates_ch2,
    std_ch1,
    std_ch2,
    inuse_ch1,
    inuse_ch2,
    pos,
):
    """Build ALC text from explicit lists of gates/std/inuse per channel."""

    idx_line = field_indices.get("Line", -1)
    idx_date = field_indices.get("Date", -1)
    idx_time = field_indices.get("Time", -1)
    idx_dem = field_indices.get("Topography", -1)
    idx_height = field_indices.get("TxAltitude", -1)
    idx_e = field_indices.get("UTMX", -1)
    idx_n = field_indices.get("UTMY", -1)
    idx_mag = field_indices.get("Magnetic", -1)
    idx_plni = field_indices.get("PowerLineMonitor", -1)
    idx_misc1 = field_indices.get("Misc1", -1)
    idx_misc2 = field_indices.get("Misc2", -1)
    idx_misc3 = field_indices.get("Misc3", -1)
    idx_misc4 = field_indices.get("Misc4", -1)

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
    # RxPitch / RxRoll removed
    lines_out.append(kv("Time", idx_time))
    lines_out.append(kv("Topography", idx_dem))
    lines_out.append(kv("TxAltitude", idx_height))
    # TxOffTime, TxOnTime, TxPeakTime removed
    # TxPitch, TxRoll removed
    # TxRxHoriSep, TxRxVertSep removed
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

    # InUse Ch01
    if inuse_ch1:
        for i, name in enumerate(inuse_ch1, start=1):
            lines_out.append(kv(f"InUse_Ch01_{i:02d}", pos.get(name, -1)))
    else:
        for i in range(1, len(gates_ch1) + 1):
            lines_out.append(kv(f"InUse_Ch01_{i:02d}", -1))
    lines_out.append("")

    # InUse Ch02
    if channels_number == 2:
        if inuse_ch2:
            for i, name in enumerate(inuse_ch2, start=1):
                lines_out.append(kv(f"InUse_Ch02_{i:02d}", pos.get(name, -1)))
        else:
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
        "n_inuse_ch1": len(inuse_ch1),
        "n_inuse_ch2": len(inuse_ch2),
    }

    return "\n".join(lines_out), info


def build_gate_mapping(gates, tag, pos, first_values, max_rows=10):
    rows = []
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


def build_std_mapping(stds, tag, pos, first_values, max_rows=10):
    rows = []
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


def build_inuse_mapping(inuse, tag, pos, first_values, n_gates, max_rows=10):
    rows = []
    if not inuse and n_gates == 0:
        return rows

    if inuse:
        for i, name in enumerate(inuse, start=1):
            if i > max_rows:
                break
            idx = pos.get(name, -1)
            val = first_values[idx - 1] if 1 <= idx <= len(first_values) else ""
            status = "✅ OK" if idx > 0 else "❌ MISSING"
            rows.append(
                {
                    "ALC entry": f"InUse_{tag}_{i:02d} = {idx}",
                    "XYZ column": f"{idx} → {name}" if idx > 0 else "Not mapped",
                    "First value": val,
                    "Status": status,
                }
            )
    else:
        # No InUse columns → all -1 for each gate
        for i in range(1, min(n_gates, max_rows) + 1):
            rows.append(
                {
                    "ALC entry": f"InUse_{tag}_{i:02d} = -1",
                    "XYZ column": "Not mapped",
                    "First value": "",
                    "Status": "❌ MISSING",
                }
            )

    return rows


# ---------- Streamlit app ----------

st.title("XYZ → SkyTEM .ALC builder (row-by-row + gate range spinners)")

st.write(
    "Upload a SkyTEM XYZ file. This app:\n"
    "- Auto-detects LM/HM gates, STD and InUse columns\n"
    "- Lets you edit ALC core indices with per-row spinners\n"
    "- Lets you choose **first/last gate** per channel (Ch01/Ch02), applied to Gates/STD/InUse\n"
    "- Generates an `.ALC` where **ALC index = XYZ index** (or -1 for unused)."
)

uploaded = st.file_uploader("Upload XYZ file", type=["xyz", "txt", "dat", "csv"])

system_name = st.text_input("System name in ALC", value="SkyTEM XYZ")

ch1_label_user = st.selectbox("Channel 1 type (Ch01)", ["LM", "HM"], index=0)
ch2_sel = st.selectbox("Channel 2 type (Ch02, optional)", ["None", "LM", "HM"], index=2)
ch2_label_user = None if ch2_sel == "None" else ch2_sel

max_rows_to_show = st.slider("Number of rows to show in mapping tables", 5, 60, 20)

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

            st.subheader("Detected columns (first 60)")
            st.write(cols[:60])

            # --- Detect channel layout (auto gates/std/inuse) ---
            layout_all = detect_channel_layout(cols, pos, ch1_label_user, ch2_label_user)
            channels_number = layout_all["channels_number"]
            ch1_label = layout_all["ch1_label"]
            ch2_label = layout_all["ch2_label"]
            gates_ch1_all = layout_all["gates_ch1_all"]
            gates_ch2_all = layout_all["gates_ch2_all"]
            std_ch1_all = layout_all["std_ch1_all"]
            std_ch2_all = layout_all["std_ch2_all"]
            inuse_ch1_all = layout_all["inuse_ch1_all"]
            inuse_ch2_all = layout_all["inuse_ch2_all"]

            st.subheader("Detected channels")
            st.json(layout_all)

            # --- Auto core indices from header names ---
            def auto_idx(name):
                return pos.get(name, -1)

            auto_field_indices = {
                "Line": auto_idx("Line"),
                "Date": auto_idx("Date"),
                "Time": auto_idx("Time"),
                "Topography": auto_idx("DEM"),
                "TxAltitude": auto_idx("Height"),
                "UTMX": auto_idx("E"),
                "UTMY": auto_idx("N"),
                "Magnetic": auto_idx("TMI"),
                "PowerLineMonitor": auto_idx("PLNI"),
                "Misc1": -1,
                "Misc2": -1,
                "Misc3": -1,
                "Misc4": -1,
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
            if channels_number == 2:
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

            # --- Gate range spinners per channel ---
            st.markdown("---")
            st.subheader("Gate range per channel (applies to Gates / STD / InUse)")

            # Channel 1
            n_g1 = len(gates_ch1_all)
            if n_g1 > 0:
                st.markdown(f"**Channel 1 – {ch1_label}** (detected {n_g1} gates)")
                g1c1, g1c2 = st.columns(2)
                with g1c1:
                    gate_start_ch1 = st.number_input(
                        "Ch01 first gate #",
                        min_value=1,
                        max_value=n_g1,
                        value=1,
                        step=1,
                        key="gate_start_ch1",
                    )
                with g1c2:
                    gate_end_ch1 = st.number_input(
                        "Ch01 last gate #",
                        min_value=gate_start_ch1,
                        max_value=n_g1,
                        value=n_g1,
                        step=1,
                        key="gate_end_ch1",
                    )
            else:
                gate_start_ch1 = 1
                gate_end_ch1 = 0
                st.info("No gates detected for Channel 1.")

            # Channel 2
            n_g2 = len(gates_ch2_all)
            if channels_number == 2 and n_g2 > 0:
                st.markdown(f"**Channel 2 – {ch2_label}** (detected {n_g2} gates)")
                g2c1, g2c2 = st.columns(2)
                with g2c1:
                    gate_start_ch2 = st.number_input(
                        "Ch02 first gate #",
                        min_value=1,
                        max_value=n_g2,
                        value=1,
                        step=1,
                        key="gate_start_ch2",
                    )
                with g2c2:
                    gate_end_ch2 = st.number_input(
                        "Ch02 last gate #",
                        min_value=gate_start_ch2,
                        max_value=n_g2,
                        value=n_g2,
                        step=1,
                        key="gate_end_ch2",
                    )
            else:
                gate_start_ch2 = 1
                gate_end_ch2 = 0

            # Apply ranges (same range for gates/std/inuse per channel)
            gates_ch1 = slice_by_range(gates_ch1_all, gate_start_ch1, gate_end_ch1)
            gates_ch2 = slice_by_range(gates_ch2_all, gate_start_ch2, gate_end_ch2)

            std_ch1 = slice_by_range(std_ch1_all, gate_start_ch1, gate_end_ch1)
            std_ch2 = slice_by_range(std_ch2_all, gate_start_ch2, gate_end_ch2)

            inuse_ch1 = slice_by_range(inuse_ch1_all, gate_start_ch1, gate_end_ch1)
            inuse_ch2 = slice_by_range(inuse_ch2_all, gate_start_ch2, gate_end_ch2)

            # --- Build ALC with these indices & ranges ---
            alc_text, info = build_alc_text(
                system_name=system_name,
                field_indices=field_indices,
                current_ch1_idx=current_ch1_idx,
                current_ch2_idx=current_ch2_idx,
                channels_number=channels_number,
                ch1_label=ch1_label,
                ch2_label=ch2_label,
                gates_ch1=gates_ch1,
                gates_ch2=gates_ch2,
                std_ch1=std_ch1,
                std_ch2=std_ch2,
                inuse_ch1=inuse_ch1,
                inuse_ch2=inuse_ch2,
                pos=pos,
            )

            st.subheader("Channel summary")
            st.json(info)

            # --- Gate / STD / InUse mapping based on selected ranges ---
            st.subheader(f"Gate mapping (Ch01 – {ch1_label})")
            gate_ch1_rows = build_gate_mapping(
                gates_ch1, "Ch01", pos, first_values, max_rows=max_rows_to_show
            )
            if gate_ch1_rows:
                st.table(pd.DataFrame(gate_ch1_rows))
            else:
                st.info("No gates selected for Channel 1.")

            if channels_number == 2:
                st.subheader(f"Gate mapping (Ch02 – {ch2_label})")
                gate_ch2_rows = build_gate_mapping(
                    gates_ch2, "Ch02", pos, first_values, max_rows=max_rows_to_show
                )
                if gate_ch2_rows:
                    st.table(pd.DataFrame(gate_ch2_rows))
                else:
                    st.info("No gates selected for Channel 2.")

            st.subheader(f"STD mapping (Ch01 – {ch1_label})")
            std_ch1_rows = build_std_mapping(
                std_ch1, "Ch01", pos, first_values, max_rows=max_rows_to_show
            )
            if std_ch1_rows:
                st.table(pd.DataFrame(std_ch1_rows))
            else:
                st.info("No STD columns (or no STD in selected gate range) for Channel 1.")

            if channels_number == 2:
                st.subheader(f"STD mapping (Ch02 – {ch2_label})")
                std_ch2_rows = build_std_mapping(
                    std_ch2, "Ch02", pos, first_values, max_rows=max_rows_to_show
                )
                if std_ch2_rows:
                    st.table(pd.DataFrame(std_ch2_rows))
                else:
                    st.info("No STD columns (or no STD in selected gate range) for Channel 2.")

            st.subheader(f"InUse mapping (Ch01 – {ch1_label})")
            inuse_ch1_rows = build_inuse_mapping(
                inuse_ch1, "Ch01", pos, first_values, n_gates=len(gates_ch1), max_rows=max_rows_to_show
            )
            if inuse_ch1_rows:
                st.table(pd.DataFrame(inuse_ch1_rows))
            else:
                st.info("No InUse columns (ALC InUse defaults to -1) for Channel 1.")

            if channels_number == 2:
                st.subheader(f"InUse mapping (Ch02 – {ch2_label})")
                inuse_ch2_rows = build_inuse_mapping(
                    inuse_ch2, "Ch02", pos, first_values, n_gates=len(gates_ch2), max_rows=max_rows_to_show
                )
                if inuse_ch2_rows:
                    st.table(pd.DataFrame(inuse_ch2_rows))
                else:
                    st.info("No InUse columns (ALC InUse defaults to -1) for Channel 2.")

            # --- ALC preview + download ---
            st.subheader("Preview of generated .ALC")
            lines_out = alc_text.splitlines()
            if len(lines_out) > 200:
                st.code("\n".join(lines_out[:200]) + "\n...", language="text")
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

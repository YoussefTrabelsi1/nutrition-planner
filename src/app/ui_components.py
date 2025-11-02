from __future__ import annotations
from typing import Dict, List, Tuple
import math
import pandas as pd
import streamlit as st


def safe_df_display(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, use_container_width=True)
    except TypeError:
        try:
            st.dataframe(df)
        except TypeError:
            st.table(df)


def unit_from_key(key: str) -> str:
    if key.endswith("_g"): return "g"
    if key.endswith("_mg"): return "mg"
    if key.endswith("_mcg"): return "mcg"
    return ""


def kpi_strip(consumed: Dict[str, float], targets: Dict[str, float], order: List[Tuple[str, str]]) -> None:
    cols = st.columns(len(order))
    for i, (label, key) in enumerate(order):
        val = float(consumed.get(key, 0.0))
        tgt = max(float(targets.get(key, 0.0)), 1e-9)
        pct = int(min(999, round(100 * val / tgt)))
        with cols[i]:
            st.metric(label, f"{val:.0f} / {tgt:.0f}", f"{pct}% of target")


def progress_row(consumed: Dict[str, float], targets: Dict[str, float], keys: List[str]) -> None:
    """Version-safe progress bars. Older Streamlit doesn't support `text=`."""
    cols = st.columns(len(keys))
    for i, k in enumerate(keys):
        val = float(consumed.get(k, 0.0))
        tgt = max(float(targets.get(k, 0.0)), 1e-9)
        pct = min(1.2, val / tgt)  # cap at 120%
        label = f"{k.capitalize()} {int(pct*100)}%"
        with cols[i]:
            try:
                # Newer Streamlit
                st.progress(pct, text=label)  # type: ignore[call-arg]
            except TypeError:
                # Older Streamlit: no `text` arg
                st.caption(label)
                st.progress(pct)
            except Exception:
                # Last resort
                st.caption(label)


def card_grid(foods_rows: List[Dict[str, str]], cols: int = 4) -> None:
    if not foods_rows:
        st.info("No foods match your filters. Try relaxing the search.")
        return
    rows = (len(foods_rows) + cols - 1) // cols
    idx = 0
    for _ in range(rows):
        cols_list = st.columns(cols)
        for col in cols_list:
            if idx >= len(foods_rows):
                break
            f = foods_rows[idx]; idx += 1
            with col:
                st.markdown(f"**{f['name']}**")
                st.caption(f"{f['serving']} • {f['kcal']} kcal")
                st.text(f"P {f['P']}g • C {f['C']}g • F {f['F']}g")
                why = f.get("why")
                if why:
                    st.caption(f"Why: {why}")
                st.button(f"➕ Add {f['name']}", key=f"add_{f['key']}", on_click=f['on_click'])

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import streamlit as st


@dataclass
class SessionPlan:
    grams_by_food: Dict[str, float] = field(default_factory=dict)

    def add(self, food_name: str, grams: float) -> None:
        if grams <= 0:
            return
        self.grams_by_food[food_name] = self.grams_by_food.get(food_name, 0.0) + grams

    def remove(self, food_name: str) -> None:
        if food_name in self.grams_by_food:
            del self.grams_by_food[food_name]

    def clear(self) -> None:
        self.grams_by_food.clear()


def get_session_plan() -> SessionPlan:
    if "session_plan" not in st.session_state:
        st.session_state["session_plan"] = SessionPlan()
    return st.session_state["session_plan"]


def set_flash(msg: str, level: str = "info") -> None:
    # Version-safe “toast”
    key = "flash_message"
    st.session_state[key] = (msg, level)


def show_flash() -> None:
    key = "flash_message"
    if key in st.session_state:
        msg, level = st.session_state.pop(key)
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "error":
            st.error(msg)
        else:
            st.info(msg)

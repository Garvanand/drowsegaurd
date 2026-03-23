import json
import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def _read_csv_safe(path):
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return None


def render_analytics():
    sessions_root = pathlib.Path("sessions")
    if not sessions_root.exists() or not any(p.is_dir() for p in sessions_root.iterdir()):
        st.info("No sessions recorded yet. Run a detection session first.")
        return

    session_dirs = sorted([p for p in sessions_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    session_names = [p.name for p in session_dirs]
    selected_session = st.selectbox("Select session", session_names)
    selected_path = sessions_root / selected_session

    events_df = _read_csv_safe(selected_path / "events.csv")
    metrics_df = _read_csv_safe(selected_path / "metrics.csv")

    if events_df is None:
        st.warning("events.csv is unreadable for the selected session.")
        events_df = pd.DataFrame()
    if metrics_df is None:
        st.warning("metrics.csv is unreadable for the selected session.")
        metrics_df = pd.DataFrame()

    total_events = int(len(events_df)) if not events_df.empty else 0
    duration = int(len(metrics_df) * 30 / 30) if not metrics_df.empty else 0
    peak_fatigue = float(metrics_df["fatigue_score"].max()) if (not metrics_df.empty and "fatigue_score" in metrics_df.columns) else 0.0
    snapshots_dir = selected_path / "snapshots"
    total_snapshots = len(list(snapshots_dir.glob("*.jpg"))) if snapshots_dir.exists() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", total_events)
    c2.metric("Session duration (s)", duration)
    c3.metric("Peak fatigue score", f"{peak_fatigue:.1f}")
    c4.metric("Total snapshots", total_snapshots)

    if not metrics_df.empty:
        if "timestamp" in metrics_df.columns:
            try:
                metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])
            except Exception:
                pass

        if "fatigue_score" in metrics_df.columns:
            fig_fatigue = px.line(metrics_df, x="timestamp" if "timestamp" in metrics_df.columns else metrics_df.index, y="fatigue_score", title="fatigue score over session")
            fig_fatigue.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.12, line_width=0)
            fig_fatigue.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.12, line_width=0)
            st.plotly_chart(fig_fatigue, use_container_width=True)

        if all(col in metrics_df.columns for col in ["ear", "perclos"]):
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            x_axis = metrics_df["timestamp"] if "timestamp" in metrics_df.columns else metrics_df.index
            fig_dual.add_trace(go.Scatter(x=x_axis, y=metrics_df["ear"], name="EAR"), secondary_y=False)
            fig_dual.add_trace(go.Scatter(x=x_axis, y=metrics_df["perclos"], name="PERCLOS"), secondary_y=True)
            fig_dual.update_layout(title="EAR and PERCLOS over session")
            fig_dual.update_yaxes(title_text="EAR", secondary_y=False)
            fig_dual.update_yaxes(title_text="PERCLOS", secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)

        if all(col in metrics_df.columns for col in ["pitch", "roll", "fatigue_score"]):
            fig_pose = px.scatter(metrics_df, x="pitch", y="roll", color="fatigue_score", title="head pose distribution")
            st.plotly_chart(fig_pose, use_container_width=True)

    if not events_df.empty:
        if "event_type" in events_df.columns:
            counts = events_df["event_type"].value_counts().reset_index()
            counts.columns = ["event_type", "count"]
            fig_events = px.bar(counts, x="event_type", y="count", title="event frequency by type")
            st.plotly_chart(fig_events, use_container_width=True)

        if "timestamp" in events_df.columns:
            try:
                events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
                events_show = events_df.sort_values("timestamp", ascending=False).head(50)
            except Exception:
                events_show = events_df.tail(50)
        else:
            events_show = events_df.tail(50)
        st.dataframe(events_show, use_container_width=True)

    if snapshots_dir.exists():
        snapshots = sorted(list(snapshots_dir.glob("*.jpg")), reverse=True)
        if snapshots:
            st.subheader("Event snapshots")
            grid = st.columns(4)
            for i, img_path in enumerate(snapshots[:12]):
                grid[i % 4].image(str(img_path), use_column_width=True)

    st.caption(json.dumps({"session": selected_session, "snapshots": total_snapshots}))

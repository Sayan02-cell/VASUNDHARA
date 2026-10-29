import io
import math
import time

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gpxpy

PAGE_TITLE = "Elevation Profile"
PAGE_ICON = "🏔️"
API_BASE = "https://api.open-elevation.com/api/v1/lookup"
CHUNK_SIZE = 250
REQUEST_DELAY = 0.4
TIMEOUT = 30

METRIC_LABELS = {"distance": "Distance (km)", "elevation": "Elevation (m)"}
IMPERIAL_LABELS = {"distance": "Distance (mi)", "elevation": "Elevation (ft)"}

M_PER_KM = 1000.0
M_PER_MI = 1609.344
M_PER_FT = 0.3048


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def parse_gpx(file):
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                if pt.latitude is not None and pt.longitude is not None:
                    points.append((pt.latitude, pt.longitude))
    for route in gpx.routes:
        for pt in route.points:
            if pt.latitude is not None and pt.longitude is not None:
                points.append((pt.latitude, pt.longitude))
    if not points:
        for wp in gpx.waypoints:
            if wp.latitude is not None and wp.longitude is not None:
                points.append((wp.latitude, wp.longitude))
    return points


def manual_points_to_list(text):
    points = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                points.append((lat, lon))
            except ValueError:
                continue
    return points


def fetch_elevations(points):
    elevations = [None] * len(points)
    total = len(points)
    if total == 0:
        return elevations

    progress = st.progress(0, text="Querying elevation API...")
    status = st.empty()

    for start in range(0, total, CHUNK_SIZE):
        chunk = points[start : start + CHUNK_SIZE]
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in chunk]}
        try:
            resp = requests.post(API_BASE, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            for i, res in enumerate(results):
                idx = start + i
                if idx < total:
                    elevations[idx] = res.get("elevation")
        except requests.RequestException as exc:
            status.warning(f"API request failed for chunk starting at {start}: {exc}")

        done = min(start + CHUNK_SIZE, total)
        progress.progress(done / total, text=f"Querying elevation API... {done}/{total}")
        if start + CHUNK_SIZE < total:
            time.sleep(REQUEST_DELAY)

    progress.empty()
    status.empty()
    return elevations


def compute_profile(points, elevations, units):
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    dist_m = [0.0]
    for i in range(1, len(points)):
        dist_m.append(dist_m[-1] + haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i]))

    valid = [(d, e) for d, e in zip(dist_m, elevations) if e is not None]
    if not valid:
        return None

    dist = [v[0] for v in valid]
    elev = [v[1] for v in valid]

    gain = 0.0
    loss = 0.0
    for i in range(1, len(elev)):
        diff = elev[i] - elev[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss += -diff

    max_elev = max(elev)
    min_elev = min(elev)
    total_dist = dist[-1]

    if units == "metric":
        dist_vals = [d / M_PER_KM for d in dist]
        x_label = METRIC_LABELS["distance"]
        y_label = METRIC_LABELS["elevation"]
        dist_suffix = " km"
        elev_suffix = " m"
        total_dist_disp = total_dist / M_PER_KM
        gain_disp = gain
        loss_disp = loss
        max_disp = max_elev
        min_disp = min_elev
    else:
        dist_vals = [d / M_PER_MI for d in dist]
        x_label = IMPERIAL_LABELS["distance"]
        y_label = IMPERIAL_LABELS["elevation"]
        dist_suffix = " mi"
        elev_suffix = " ft"
        total_dist_disp = total_dist / M_PER_MI
        gain_disp = gain / M_PER_FT
        loss_disp = loss / M_PER_FT
        max_disp = max_elev / M_PER_FT
        min_disp = min_elev / M_PER_FT

    return {
        "distance": dist_vals,
        "elevation": elev,
        "x_label": x_label,
        "y_label": y_label,
        "dist_suffix": dist_suffix,
        "elev_suffix": elev_suffix,
        "total_distance": total_dist_disp,
        "gain": gain_disp,
        "loss": loss_disp,
        "max_elevation": max_disp,
        "min_elevation": min_disp,
        "start_elevation": elev[0],
        "end_elevation": elev[-1],
        "num_points": len(elev),
    }


def render_profile_chart(profile, units, figsize=(12, 6), dpi=150):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    dist = profile["distance"]
    elev = profile["elevation"]

    ax.plot(dist, elev, color="#2E86C1", linewidth=2.0, solid_capstyle="round")
    ax.fill_between(dist, elev, color="#2E86C1", alpha=0.18)

    ax.set_xlabel(profile["x_label"], fontsize=12, fontweight="bold", color="#333333")
    ax.set_ylabel(profile["y_label"], fontsize=12, fontweight="bold", color="#333333")
    ax.set_title("Elevation Profile", fontsize=16, fontweight="bold", color="#1a1a1a", pad=15)

    ax.grid(True, which="major", color="#cccccc", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color="#dddddd", linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

    ax.tick_params(axis="both", which="major", labelsize=10, colors="#555555")
    ax.tick_params(axis="both", which="minor", labelsize=8, colors="#888888")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")

    y_min, y_max = min(elev), max(elev)
    y_range = y_max - y_min
    if y_range == 0:
        y_range = abs(y_max) if y_max != 0 else 1.0
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.08 * y_range)

    x_min, x_max = min(dist), max(dist)
    x_range = x_max - x_min
    if x_range == 0:
        x_range = 1.0
    ax.set_xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)

    stats_text = (
        f"Max Elev: {profile['max_elevation']:,.1f}{profile['elev_suffix']}\n"
        f"Min Elev: {profile['min_elevation']:,.1f}{profile['elev_suffix']}\n"
        f"Gain: {profile['gain']:,.1f}{profile['elev_suffix']}\n"
        f"Loss: {profile['loss']:,.1f}{profile['elev_suffix']}\n"
        f"Distance: {profile['total_distance']:,.2f}{profile['dist_suffix']}"
    )

    ax.text(
        0.985,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#1a1a1a",
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor="#2E86C1",
            linewidth=1.4,
            alpha=0.95,
        ),
    )

    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    st.title(f"{PAGE_ICON} Elevation Profile Generator")
    st.markdown(
        "Generate a Google Earth Pro-style elevation profile from a GPX track or manual coordinates. "
        "Elevation data is fetched from the public Open-Elevation API."
    )

    with st.sidebar:
        st.header("Input Source")
        source = st.radio("Choose input method:", ["GPX File Upload", "Manual Coordinates"], index=0)

        units = st.selectbox("Distance / Elevation units:", ["Metric (km, m)", "Imperial (mi, ft)"], index=0)
        unit_key = "metric" if units.startswith("Metric") else "imperial"

        st.header("Export")
        export_dpi = st.selectbox("PNG export DPI:", [150, 200, 300, 600], index=2)

    points = None

    if source == "GPX File Upload":
        uploaded = st.file_uploader("Upload a .gpx track file", type=["gpx"])
        if uploaded is not None:
            try:
                points = parse_gpx(uploaded)
                st.success(f"Parsed {len(points)} track points from GPX file.")
            except Exception as exc:
                st.error(f"Failed to parse GPX file: {exc}")
    else:
        st.subheader("Manual Coordinates")
        st.markdown("Enter latitude and longitude pairs, one per line (e.g. `34.0522, -118.2437`).")
        coord_text = st.text_area(
            "Coordinates",
            height=200,
            placeholder="34.0522, -118.2437\n34.1000, -118.1000\n34.1500, -118.0000",
        )
        if st.button("Sample Path", type="primary"):
            if coord_text.strip():
                points = manual_points_to_list(coord_text)
                st.success(f"Parsed {len(points)} coordinate pairs.")
            else:
                st.warning("Please enter at least one coordinate pair.")

    if points is None or len(points) == 0:
        st.info("Provide a GPX file or coordinates to generate the elevation profile.")
        return

    with st.expander("Preview Points", expanded=False):
        df_preview = pd.DataFrame(points, columns=["latitude", "longitude"])
        st.dataframe(df_preview.head(50), use_container_width=True)
        st.caption(f"Total points: {len(points)}")

    if st.button("Generate Elevation Profile", type="primary"):
        with st.spinner("Fetching elevation data..."):
            elevations = fetch_elevations(points)

        valid_count = sum(1 for e in elevations if e is not None)
        st.info(f"Retrieved elevations for {valid_count}/{len(points)} points.")

        profile = compute_profile(points, elevations, unit_key)
        if profile is None:
            st.error("No valid elevation data was returned. Cannot generate profile.")
            return

        st.session_state["profile"] = profile
        st.session_state["units"] = unit_key

    profile = st.session_state.get("profile")
    if profile is None:
        return

    st.subheader("Elevation Profile")
    fig = render_profile_chart(profile, st.session_state.get("units", "metric"))
    st.pyplot(fig, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max Elevation", f"{profile['max_elevation']:,.1f}{profile['elev_suffix']}")
    c2.metric("Min Elevation", f"{profile['min_elevation']:,.1f}{profile['elev_suffix']}")
    c3.metric("Elevation Gain", f"{profile['gain']:,.1f}{profile['elev_suffix']}")
    c4.metric("Elevation Loss", f"{profile['loss']:,.1f}{profile['elev_suffix']}")
    c5.metric("Distance", f"{profile['total_distance']:,.2f}{profile['dist_suffix']}")

    st.subheader("Export")
    png_bytes = fig_to_png_bytes(fig, dpi=export_dpi)
    st.download_button(
        label="Download Profile as PNG",
        data=png_bytes,
        file_name="elevation_profile.png",
        mime="image/png",
        type="primary",
    )

    st.subheader("Raw Profile Data")
    df = pd.DataFrame(
        {
            "distance": profile["distance"],
            "elevation": profile["elevation"],
        }
    )
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Profile as CSV",
        data=csv,
        file_name="elevation_profile.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

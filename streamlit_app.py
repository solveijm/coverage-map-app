import streamlit as st
import polars as pl
import pydeck as pdk

REQUIRED_COLS = {
    "lat": ["Latitude", "latitude", "LAT", "lat"],
    "lon": ["Longitude", "longitude", "LON", "lon"],
    "rsrp": ["RSRP (dBm)", "RSRP", "rsrp_dbm", "rsrp"]
}


def find_col(df: pl.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

def coerce_numeric(df: pl.DataFrame, col: str, as_float: bool = True) -> pl.Series:
    try:
        if as_float:
            return df[col].cast(pl.Float64, strict=False)
        return df[col].cast(pl.Int64, strict=False)
    except Exception:
        return pl.col(col).str.replace_all(",", ".").cast(pl.Float64, strict=False)

def validate_and_prepare(df: pl.DataFrame) -> pl.DataFrame:
    lat_col = find_col(df, REQUIRED_COLS["lat"])
    lon_col = find_col(df, REQUIRED_COLS["lon"])
    rsrp_col = find_col(df, REQUIRED_COLS["rsrp"])

    missing = [k for k, v in zip(["Latitude", "Longitude", "RSRP (dBm)"], [lat_col, lon_col, rsrp_col]) if v is None]
    if missing:
        raise ValueError(
            "Missing required columns. Expected columns (case-insensitive) like: "
            "Latitude, Longitude, and RSRP (dBm)."
        )

    df = df.rename({lat_col: "lat", lon_col: "lon", rsrp_col: "RSRP (dBm)"})

    # --- drop nulls / invalid numeric values ---
    df = df.drop_nulls(["lat", "lon", "RSRP (dBm)"])

    # --- plausible ranges ---
    df = df.filter(
        (pl.col("lat").is_between(-90, 90)) &
        (pl.col("lon").is_between(-180, 180))
    )

    if df.height == 0:
        raise ValueError("No valid rows after cleaning (nulls or out-of-range values).")

    # --- normalize RSRP to [0,1] and compute RGB (red→green) ---
    # Common RSRP spans roughly −135 dBm (weak) to −85 dBm (strong).
    # We map [-135, -85] → [0, 1] and clamp.
    t = ((pl.col("RSRP (dBm)") + 135) / 50).clip(0, 1)

    df = df.with_columns(
        t.alias("t"),
        (255 * (1 - t)).round().cast(pl.Int64).alias("r"),
        (255 * t).round().cast(pl.Int64).alias("g"),
        pl.lit(0).alias("b"),
    )

    return df.select(["lat", "lon", "RSRP (dBm)", "r", "g", "b"])

def compute_view(df_pd) -> pdk.ViewState:
    # Heuristic: center on mean; zoom based on geographic spread
    lat_mean = float(df_pd["lat"].mean())
    lon_mean = float(df_pd["lon"].mean())

    lat_span = float(df_pd["lat"].max() - df_pd["lat"].min())
    lon_span = float(df_pd["lon"].max() - df_pd["lon"].min())
    span = max(lat_span, lon_span)

    # Rough mapping span→zoom (very rough, but avoids zoom=3 on dense local data)
    if span < 0.01:
        zoom = 15
    elif span < 0.1:
        zoom = 12
    elif span < 1:
        zoom = 9
    elif span < 5:
        zoom = 6
    else:
        zoom = 3

    return pdk.ViewState(latitude=lat_mean, longitude=lon_mean, zoom=zoom)

def main():
    st.title("Coverage Map Visualization")

    # Accept standard Parquet extensions ('.parquet'); users sometimes upload without extension via Streamlit
    file = st.file_uploader("Upload your cellular coverage data (Parquet format)", type=["pl"])
    if file is None:
        st.info("Upload a Parquet file containing Latitude, Longitude, and RSRP (dBm) columns.")
        return

    # --- Read parquet safely ---
    try:
        df = pl.read_parquet(file)
        if df.is_empty():
            st.error("The uploaded file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading Parquet file: {e}")
        st.stop()

    # --- Validate / prepare ---
    try:
        df = validate_and_prepare(df)
    except ValueError as ve:
        st.error(str(ve))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while preparing data: {e}")
        st.stop()

    # --- Build layer & deck ---
    try:
        df_pd = df.to_pandas()  # pydeck consumes pandas
    except Exception as e:
        st.error(f"Failed to convert to pandas for visualization: {e}")
        st.stop()

    try:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_pd,
            get_position="[lon, lat]",
            get_fill_color="[r, g, b, 180]",  # RGBA
            get_radius=30,
            pickable=True,
            auto_highlight=True,
        )

        view = compute_view(df_pd)

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"html": "RSRP: {RSRP (dBm)} dBm<br/>Lat: {lat}<br/>Lon: {lon}"},
        )

        st.caption("RSRP color scale: red = weak signal (≈−135 dBm), green = strong signal (≈−85 dBm).")
        st.pydeck_chart(deck)
    except Exception as e:
        st.error(f"Visualization error: {e}")
        st.stop()

if __name__ == "__main__":
    main()

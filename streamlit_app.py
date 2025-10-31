import streamlit as st
import polars as pl
import pydeck as pdk


st.title("Coverage Map Visualization")

file = st.file_uploader("Upload your cellular coverage data (Parquet format)", type=["pl"])
if file is not None:
    # Load & prep
    df = (
        pl.read_parquet(file)
        .rename({"Latitude": "lat", "Longitude": "lon"})
        .drop_nulls()
    )

    # Normalize RSRP from [-100, 50] -> [0, 1], clamp to be safe
    t = ((pl.col("RSRP (dBm)") + 135) / 50).clip(0, 1)


    # Compute RGB: red -> green
    df = df.with_columns(
        t.alias("t"),
        ((255 * (1 - t)).round().cast(pl.Int64)).alias("r"),
        ((255 * t).round().cast(pl.Int64)).alias("g"),
        pl.lit(0).alias("b"),
    )


    # PyDeck layer using computed colors
    df_pd = df.select(["lat", "lon", "RSRP (dBm)", "r", "g", "b"]).to_pandas()

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_pd,
        get_position="[lon, lat]",
        get_fill_color="[r, g, b, 180]",   # RGBA, 0–255
        get_radius=30,                     # tweak to taste
        pickable=True,
        auto_highlight=True,
    )

    # Center/zoom: quick heuristic
    view = pdk.ViewState(
        latitude=float(df_pd["lat"].mean()),
        longitude=float(df_pd["lon"].mean()),
        zoom=3,
    )

    # Force light mode basemap
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"html": "RSRP: {RSRP (dBm)} dBm<br/>Lat: {lat}<br/>Lon: {lon}"},
    )

    st.pydeck_chart(deck)


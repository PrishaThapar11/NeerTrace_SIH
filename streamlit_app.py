# streamlit_app.py
"""
Streamlit app for HMPI Analyzer (heavy-metal pollution indices).

Features:
- Upload CSV (same schema as seed_samples.csv) -> computes indices and displays results
- Manual single-sample entry -> compute indices immediately
- Table preview + basic Plotly bar chart for a selected sample
- Optional map (folium) if lat/lon present and folium + streamlit_folium installed
- Download processed CSV
"""
# Add to top imports
import html 
import numpy as np
import folium
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
import streamlit as st
import pandas as pd
import io
import os
from hmpi_utils import load_config, compute_indices_for_df, detect_metals_in_df

# Optional visualization imports
try:
    import plotly.express as px
except Exception:
    px = None

# optional mapping
USE_FOLIUM = False
try:
    import folium
    from streamlit_folium import st_folium
    USE_FOLIUM = True
except Exception:
    USE_FOLIUM = False


def get_color_for_pli_category(cat: str):
    """Return a hex color for a PLI category string (case-insensitive)."""
    if not cat:
        return "#757575"  # grey for unknown
    s = cat.lower()
    # explicit checks in order of priority
    if "very highly" in s or "very highly polluted" in s:
        return "#800026"  # very dark red
    if "highly" in s or ("high" in s and "pollut" in s):
        return "#d73027"  # red
    if "moderately" in s or "moderate" in s:
        return "#fdae61"  # orange
    if "unpolluted" in s or "safe" in s:
        return "#1a9850"  # green
    return "#757575"  # fallback grey

def add_legend(folium_map, title="PLI Categories"):
    """
    Add a simple, robust legend to a folium map using plain HTML inserted as an Element.
    This forces label color to black and uses inline-block squares so labels always appear.
    """
    legend_html = f"""
    <div style="
      position: fixed;
      bottom: 50px;
      left: 50px;
      width:220px;
      z-index:9999;
      font-size:13px;
      font-family: Arial, Helvetica, sans-serif;
      color: #000000;
      background-color: rgba(255,255,255,0.95);
      border:2px solid rgba(0,0,0,0.2);
      padding: 10px;
      box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:700; margin-bottom:6px;">{title}</div>
      <div style="display:flex; align-items:center; margin-bottom:4px;">
        <span style="display:inline-block;width:14px;height:14px;background:#1a9850;margin-right:8px;border:1px solid #666;"></span>
        <span style="color:#000">Unpolluted</span>
      </div>
      <div style="display:flex; align-items:center; margin-bottom:4px;">
        <span style="display:inline-block;width:14px;height:14px;background:#fdae61;margin-right:8px;border:1px solid #666;"></span>
        <span style="color:#000">Moderately Polluted</span>
      </div>
      <div style="display:flex; align-items:center; margin-bottom:4px;">
        <span style="display:inline-block;width:14px;height:14px;background:#d73027;margin-right:8px;border:1px solid #666;"></span>
        <span style="color:#000">Highly Polluted</span>
      </div>
      <div style="display:flex; align-items:center;">
        <span style="display:inline-block;width:14px;height:14px;background:#757575;margin-right:8px;border:1px solid #666;"></span>
        <span style="color:#000">Unknown</span>
      </div>
    </div>
    """
    folium_map.get_root().html.add_child(folium.Element(legend_html))


def plot_color_coded_map(df_out, default_center=None, zoom_start=12):
    """
    Create a folium map with circle markers colored by PLI category and a reliable legend.
    Returns folium.Map object.
    """
    # compute reasonable default center
    if default_center is None:
        try:
            default_center = (float(df_out["lat"].mean()), float(df_out["lon"].mean()))
        except Exception:
            default_center = (26.23, 92.79)

    m = folium.Map(location=default_center, zoom_start=zoom_start, tiles="OpenStreetMap")

    # iterate rows and add circle markers with popups
    for _, r in df_out.iterrows():
        try:
            lat = float(r.get("lat"))
            lon = float(r.get("lon"))
        except Exception:
            continue  # skip rows without valid coords

        pli_cat = r.get("PLI_category", "Unknown")
        color = get_color_for_pli_category(pli_cat)

        # build concise popup HTML
        popup_html = f"""
        <div style="font-size:13px;">
          <b>{r.get('sample_id','')}</b> - {r.get('site_name','')}<br/>
          <b>PLI:</b> {r.get('PLI', 'NA')}<br/>
          <b>Category:</b> {pli_cat}<br/>
          <small>
            As: {r.get('arsenic_ugL','NA')} µg/L<br/>
            Pb: {r.get('lead_ugL','NA')} µg/L<br/>
            Cr: {r.get('chromium_ugL','NA')} µg/L
          </small>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    # add legend last
    add_legend(m, title="PLI Categories")
    return m
# -----------------------------------------------------------------------------------

st.set_page_config(page_title="HMPI Analyzer", layout="wide")

st.title("NeerTrace (HMPI Analyzer) — Heavy Metal Pollution Indices")
st.markdown("Upload groundwater sample CSV or enter a single sample. Uses configured background values in `config.json`.")

# load config
try:
    cfg = load_config("config.json")
except Exception as e:
    st.error(f"Could not load config.json: {e}")
    st.stop()

# Sidebar: quick instructions and config display
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
- CSV should contain columns like `sample_id,date,site_name,lat,lon,arsenic_ugL,lead_ugL,...`.
- Core metals (by default) are: `arsenic_ugL, lead_ugL, cadmium_ugL, chromium_ugL, mercury_ugL`.
- Upload CSV or enter one sample manually.
"""
    )
    st.markdown("**Background values (from config.json)**")
    st.json(cfg.get("background_values", {}))

# --- Upload CSV -------------------------------------------------------------
st.header("1) Upload CSV (batch)")
uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="upload_csv")
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} rows from uploaded CSV.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# If no upload, load default seed if present
if df is None and os.path.exists("seed_samples.csv"):
    if st.button("Use demo seed_samples.csv (local)"):
        df = pd.read_csv("seed_samples.csv")
        st.success(f"Loaded demo seed_samples.csv ({len(df)} rows).")

# Compute indices if df loaded
if df is not None:
    metals_present, metals_with_bg = detect_metals_in_df(df, cfg)
    st.info(f"Metals present: {metals_present}  |  Metals used for indices: {metals_with_bg}")
    df_out = compute_indices_for_df(df, cfg)
    st.subheader("Computed results (first 10 rows)")
    st.dataframe(df_out.head(10))

    # Download button
    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed CSV", data=csv_bytes, file_name="processed_samples_with_indices.csv", mime="text/csv")

    # ----------------- Executive summary -----------------
    st.markdown("### Executive summary")
    n = len(df_out)
    n_hot = df_out[df_out["PLI"] > 2].shape[0]
    n_mod = df_out[(df_out["PLI"] > 1) & (df_out["PLI"] <= 2)].shape[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples processed", n)
    col2.metric("Highly polluted (PLI>2)", n_hot)
    col3.metric("Moderately polluted (1<PLI≤2)", n_mod)

    st.markdown("**Top 3 sites by PLI**")
    top3 = df_out.sort_values("PLI", ascending=False).head(3)[["sample_id","site_name","PLI","PLI_category"]]
    st.table(top3)

    # ----------------- nicer table (rounded) -----------------
    st.subheader("Full results (rounded for readability)")
    # format numeric columns to 3 decimals where applicable
    fmt_df = df_out.copy()
    numcols = fmt_df.select_dtypes(include=["float","int"]).columns.tolist()
    fmt_df[numcols] = fmt_df[numcols].round(3)
    st.dataframe(fmt_df, use_container_width=True)

    # ----------------- Color-coded map -----------------
    if USE_FOLIUM and ("lat" in df_out.columns and "lon" in df_out.columns):
        st.subheader("Map — color coded by PLI category")
        folium_map = plot_color_coded_map(df_out)
        st_folium(folium_map, width=900, height=520)
    else:
        st.info("Install folium and streamlit-folium to enable an interactive color-coded map: pip install folium streamlit-folium")


    # sample selector for detail view
    st.subheader("Inspect a sample")
    if "sample_id" in df_out.columns:
        sid = st.selectbox("Select sample_id", df_out["sample_id"].tolist())
        row = df_out[df_out["sample_id"] == sid].iloc[0]
        st.markdown(f"**Site:** {row.get('site_name','-')}  •  **Date:** {row.get('date','-')}")
        st.write("PLI:", row.get("PLI"), " — ", row.get("PLI_category"))
        # show CF columns
        cf_cols = [c for c in df_out.columns if c.startswith("CF_")]
        if cf_cols:
            cf_vals = row[cf_cols].to_dict()
            st.write("Contamination Factors (CF):")
            st.table(pd.DataFrame(list(cf_vals.items()), columns=["metal","CF"]).set_index("metal"))

            # bar chart of metal concentrations vs background (if plotly available)
            if px is not None:
                metals = [c.replace("CF_","") for c in cf_cols]
                concentrations = [row.get(m, None) for m in metals]
                backgrounds = [cfg.get("background_values", {}).get(m, None) for m in metals]
                plot_df = pd.DataFrame({
                    "metal": metals,
                    "concentration": concentrations,
                    "background": backgrounds
                })
                plot_df = plot_df.dropna()
                if not plot_df.empty:
                    fig = px.bar(plot_df.melt(id_vars="metal", value_vars=["concentration","background"], var_name="type", value_name="value"),
                                 x="metal", y="value", color="type", barmode="group",
                                 title="Concentration vs Background")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install plotly for charts: pip install plotly")

        # optional map
        if USE_FOLIUM and pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
            st.subheader("Location map")
            m = folium.Map(location=[float(row["lat"]), float(row["lon"])], zoom_start=12)
            folium.Marker([float(row["lat"]), float(row["lon"])], popup=str(row.get("site_name",""))).add_to(m)
            st_folium(m, width=700, height=350)
        else:
            if not USE_FOLIUM:
                st.info("Install folium and streamlit_folium to enable maps: pip install folium streamlit-folium")



      # --- Manual single-sample entry (fixed) ---------------------------------------------
st.header("2) Manual single-sample entry")

# Initialize session state holder (only once)
if "last_manual_result" not in st.session_state:
    st.session_state["last_manual_result"] = None

with st.form("manual_form"):
    sid = st.text_input("Sample ID", value="SMP_NEW")
    site = st.text_input("Site name", value="Manual_Site")
    date = st.date_input("Sample date")
    lat = st.text_input("Latitude (optional)", value="")
    lon = st.text_input("Longitude (optional)", value="")
    st.markdown("Enter metal concentrations (µg/L). Leave blank if unknown.")
    # dynamic fields from config (all metals present in config)
    all_mets = cfg.get("metal_columns", []) + cfg.get("optional_metal_columns", [])
    manual_vals = {}
    cols = st.columns(3)
    for i, m in enumerate(all_mets):
        label = m.replace("_ugL", "").upper()
        val = cols[i % 3].text_input(label, key=f"m_{m}")
        manual_vals[m] = val
    submitted = st.form_submit_button("Compute for this sample")

    if submitted:
        # build df with single row
        row = {
            "sample_id": sid,
            "date": date.isoformat(),
            "site_name": site,
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None
        }
        for m, v in manual_vals.items():
            try:
                row[m] = float(v) if v not in (None, "") else None
            except:
                row[m] = None
        df_single = pd.DataFrame([row])
        df_single_out = compute_indices_for_df(df_single, cfg)

        # store result in session state so we can access it outside the form
        st.session_state["last_manual_result"] = df_single_out

        # Show results inside the form (allowed)
        st.success("Computed indices for manual sample:")
        st.table(df_single_out.T)  # transpose so it's easy to read

# Outside the form: show download button if we have a last result
# ----------------- Show manual-sample result banner, map and download -----------------
if st.session_state.get("last_manual_result") is not None:
    df_last = st.session_state["last_manual_result"]

    # Ensure df_last is a DataFrame and non-empty
    if not isinstance(df_last, pd.DataFrame) or len(df_last) == 0:
        st.error("Stored manual result is not a non-empty DataFrame.")
    else:
        # Convert first row to a plain dict so .get() calls are safe
        row_series = df_last.iloc[0]
        # If row_series is a Series, to_dict() will work; otherwise fall back to dict()
        row = row_series.to_dict() if hasattr(row_series, "to_dict") else dict(row_series)

        # 1) Big colored banner: category + PLI
        pli_val = row.get("PLI", None)
        pli_cat = row.get("PLI_category", "Unknown")
        color = get_color_for_pli_category(pli_cat)

        # Prepare escaped strings for safe HTML insertion
        def safe_str(x, none_rep="-"):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return none_rep
            # decode bytes if present
            if isinstance(x, (bytes, bytearray)):
                try:
                    x = x.decode("utf-8", errors="replace")
                except:
                    x = str(x)
            return html.escape(str(x))

        safe_sample_id = safe_str(row.get("sample_id", ""))
        safe_site_name = safe_str(row.get("site_name", "-"))
        safe_date = safe_str(row.get("date", "-"))
        # --- HMPI: get & format (use same safe_str util)
        hmpi_val = row.get("HMPI", None)
        hmpi_cat = row.get("HMPI_category", "Unknown")

# Format HMPI numeric value nicely (2 decimals) or NA
        if hmpi_val is None or (isinstance(hmpi_val, float) and np.isnan(hmpi_val)):
         safe_hmpi_val = "NA"
        else:
             try:
              safe_hmpi_val = f"{float(hmpi_val):.2f}"
             except Exception:
              safe_hmpi_val = safe_str(hmpi_val)

        safe_hmpi_cat = safe_str(hmpi_cat)

        # PLI formatting
        if pli_val is None or (isinstance(pli_val, float) and np.isnan(pli_val)):
            safe_pli_val = "NA"
        else:
            try:
                safe_pli_val = f"{float(pli_val):.2f}"
            except:
                safe_pli_val = safe_str(pli_val)
        safe_pli_cat = safe_str(pli_cat)

        # Validate color (simple check for hex or word); fallback if suspicious
        import re
        if not re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$|^[a-zA-Z]+$", str(color)):
            color = "#999"

        # build HTML banner (large, centered) using escaped values
        
        banner_html = f"""
        <div style="border-radius:8px; padding:16px; margin:6px 0;
            background: linear-gradient(90deg, rgba(255,255,255,0.95), rgba(250,250,250,0.95));
            border-left:6px solid {color};">
         <div style="display:flex; align-items:center; justify-content:space-between;">
          <div style="color:#0a3780;">
           <div style="font-size:20px; font-weight:700;">
             <span style="color:#0a3780;">Manual sample:</span>
             <span style="color:#333; margin-left:8px;">{safe_sample_id}</span>
           </div>
           <div style="font-size:14px; color:#0a3780;">{safe_site_name} • {safe_date}</div>
         </div>
         <div style="text-align:right;">
           <div style="font-size:18px; font-weight:800; color:#0a3780;">
             <span>PLI:</span>
             <span style="color:#0a3780; margin-left:6px;">{safe_pli_val}</span>
           </div>
           <div style="font-size:16px; font-weight:700; margin-top:6px;">
            <span style="background:{color}; color:#fff; padding:4px 8px; border-radius:6px; font-weight:700;">
            {safe_pli_cat}
            </span>
                  <!-- NEW: HMPI display -->
           <div style="font-size:14px; margin-top:10px; text-align:right; color:#222;">
             <span style="font-weight:700;">HMPI:</span>
              <span style="margin-left:8px;">{safe_hmpi_val}</span>
                <span style="background:#444; color:#fff; padding:3px 7px; border-radius:6px; margin-left:8px; font-size:12px;">
                {safe_hmpi_cat}
             </span>
            </div>
         </div>
       </div>
     </div>
</div>
"""

        st.markdown(banner_html, unsafe_allow_html=True)

        # 2) Show a small two-column area: left = popup table, right = mini map
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("Manual sample details")

            # Create a display-safe DataFrame for st.table and CSV (so Arrow serialization doesn't fail)
            df_display = df_last.copy()

            # Convert object columns to safe strings (decode bytes, replace NaN with empty string)
            for c in df_display.columns:
                if df_display[c].dtype == object:
                    def _to_safe_string(v):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return ""
                        if isinstance(v, (bytes, bytearray)):
                            try:
                                v = v.decode("utf-8", errors="replace")
                            except:
                                v = str(v)
                        return str(v)
                    df_display[c] = df_display[c].apply(_to_safe_string)

            # show the computed dataframe transposed (readable)
            st.table(df_display.T)

        with c2:
            # show map if coords available
            lat_ok = pd.notna(row.get("lat"))
            lon_ok = pd.notna(row.get("lon"))
            if USE_FOLIUM and lat_ok and lon_ok:
                try:
                    lat = float(row.get("lat"))
                    lon = float(row.get("lon"))
                    m_single = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")

                    # Use escaped values for the popup
                    popup_html = f"<b>{safe_sample_id}</b><br/>{safe_site_name}<br/>PLI: {safe_pli_val}"
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=10,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        popup=folium.Popup(popup_html, max_width=250)
                    ).add_to(m_single)

                    # optional: small legend for single-sample map, using escaped category
                    legend_html = f"""
                    <div style="
                      position: fixed;
                      bottom: 10px;
                      left: 10px;
                      width:140px;
                      z-index:9999;
                      font-size:12px;
                      background-color: rgba(255,255,255,0.95);
                      border:1px solid rgba(0,0,0,0.1);
                      padding:6px;
                    ">
                      <b>Category</b><br>
                      <span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #666;"></span>
                      <span style="color:#000">{safe_pli_cat}</span>
                    </div>
                    """
                    m_single.get_root().html.add_child(folium.Element(legend_html))
                    st_folium(m_single, width=350, height=300)
                except Exception as e:
                    st.error(f"Could not render single-sample map: {e}")
            else:
                st.info("Latitude/Longitude missing or folium not installed; map not shown.")

        # 3) Download button (kept outside the form)
        # Use df_display (safe strings) for download to avoid pyarrow/serialization issues
        csv_bytes = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download this sample result (CSV)",
            data=csv_bytes,
            file_name=f"{row.get('sample_id','manual')}_with_indices.csv",
            mime="text/csv"
        )
# --------------------------------------------------------------------------------------


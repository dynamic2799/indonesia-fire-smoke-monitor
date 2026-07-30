
import io
import os
import re
from ftplib import FTP, error_perm
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Indonesia Fire & Smoke Monitor",
    page_icon="🔥",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}
.main-header {
    background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 45%, #b91c1c 100%);
    border-radius: 18px;
    padding: 1.15rem 1.35rem;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.main-header h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 700;
}
.main-header p {
    margin: 0.3rem 0 0 0;
    opacity: 0.95;
    font-size: 0.98rem;
}
.operational-badge {
    display: inline-block;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}
.badge-green { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
.kpi-card {
    border-radius: 16px;
    padding: 1rem 1rem 0.95rem 1rem;
    background: white;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    min-height: 115px;
}
.kpi-label {
    font-size: 0.88rem;
    color: #4b5563;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.kpi-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
}
.kpi-sub {
    margin-top: 0.4rem;
    font-size: 0.82rem;
    color: #6b7280;
}
.panel-card {
    border-radius: 18px;
    background: white;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    padding: 1rem 1rem 0.75rem 1rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.35rem;
}
.panel-subtitle {
    font-size: 0.86rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.legend-card {
    border-radius: 16px;
    padding: 0.9rem 1rem;
    background: #f8fafc;
    border: 1px solid rgba(0,0,0,0.06);
    margin-top: 0.5rem;
}
.legend-title {
    font-weight: 700;
    font-size: 0.92rem;
    margin-bottom: 0.4rem;
}
.legend-item {
    font-size: 0.85rem;
    color: #374151;
    margin-bottom: 0.25rem;
}
.archive-note {
    font-size: 0.84rem;
    color: #6b7280;
}
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.block-container { padding-top: 0.55rem; padding-bottom: 0.7rem; max-width: 100%; }
header[data-testid="stHeader"] { height: 2.3rem; }
div[data-testid="stSidebar"] { min-width: 245px; max-width: 280px; }
div[data-testid="stSidebar"] .block-container { padding-top: 0.7rem; }
div[data-testid="stTabs"] button { padding-top: 0.3rem; padding-bottom: 0.3rem; font-size: 0.88rem; }
.main-header { padding: 0.75rem 1rem; margin-bottom: 0.45rem; border-radius: 12px; }
.main-header h1 { font-size: 1.48rem; }
.main-header p { font-size: 0.82rem; }
.kpi-card { min-height: 84px; padding: 0.65rem 0.75rem; border-radius: 11px; }
.kpi-label { font-size: 0.74rem; }
.kpi-value { font-size: 1.1rem; }
.kpi-sub { font-size: 0.7rem; }
div[data-testid="stPlotlyChart"] { margin-top: -0.35rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>Indonesia Fire & Smoke Monitor</h1>
    <p>Dashboard operasional untuk pemantauan hotspot dan indikasi asap berbasis Himawari Smoke RGB.</p>
    <p>Indonesia Coverage dan Area Khusus Pemantauan Kab. Berau</p>
    <p>Created by ulil.hidayat@bmkg.go.id & Tim BMKG Berau</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================
AREA_PRESETS = {
    "Indonesia": {
        "lon_min": 94.5,
        "lon_max": 141.5,
        "lat_min": -11.5,
        "lat_max": 8.0,
    },
    "Berau": {
        "lon_min": 116.0,
        "lon_max": 119.5,
        "lat_min": 0.5,
        "lat_max": 2.8,
    },
}

SMOKE_RGB = {
    "r_min": 0.0,
    "r_max": 110.0,
    "r_gamma": 1.8,
    "g_min": 0.0,
    "g_max": 100.0,
    "g_gamma": 1.0,
    "b_min": 8.0,
    "b_max": 60.0,
    "b_gamma": 3.0,
}


FTP_SERVER = "ftp.ptree.jaxa.jp"
DEFAULT_AHI_SUFFIX = "02801_02401"
DEFAULT_HOTSPOT_SUFFIX = "06001_06001"

DEFAULT_STATE = {
    "last_processed": {},
    "latest_seen_ahi_time": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SECURE CREDENTIALS
# ============================================================
def get_secret(name, default=""):
    """
    Ambil credential dari Streamlit Secrets.
    Jika dijalankan lokal, gunakan environment variable sebagai fallback.
    """
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


PTREE_USERNAME = get_secret("PTREE_USERNAME", "")
PTREE_PASSWORD = get_secret("PTREE_PASSWORD", "")


# ============================================================
# FILE HELPERS
# ============================================================
AHI_PATTERN = re.compile(
    r"NC_(H\d{2})_(\d{8})_(\d{4})_R21_FLDK\.(\d{5}_\d{5})\.nc$"
)

HOTSPOT_PATTERN = re.compile(
    r"H\d{2}_(\d{8})_(\d{4})_L2WLF010_FLDK\.(\d{5}_\d{5})\.csv$"
)


def parse_ahi_time(path):
    match = AHI_PATTERN.match(Path(path).name)
    if match is None:
        return None
    return datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M")


def parse_hotspot_time(path):
    match = HOTSPOT_PATTERN.match(Path(path).name)
    if match is None:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")


def find_latest_ahi(folder):
    folder = Path(folder)
    if not folder.exists():
        return None, None

    candidates = []
    for path in folder.rglob("*.nc"):
        timestamp = parse_ahi_time(path)
        if timestamp is not None:
            candidates.append((timestamp, path))

    if not candidates:
        return None, None

    return max(candidates, key=lambda item: item[0])


def find_matching_hotspot(folder, ahi_time, fallback_hours=2):
    """Pilih hotspot lokal terbaru dalam rentang maksimal 2 jam sebelum AHI."""
    folder = Path(folder)
    if not folder.exists():
        return None

    earliest_time = ahi_time - timedelta(hours=fallback_hours)
    candidates = []

    for path in folder.rglob("*.csv"):
        timestamp = parse_hotspot_time(path)
        if timestamp is None:
            continue
        if earliest_time <= timestamp <= ahi_time:
            candidates.append((timestamp, path))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def archive_path(root, area_name, obs_time):
    directory = (
        Path(root)
        / area_name
        / obs_time.strftime("%Y")
        / obs_time.strftime("%m")
        / obs_time.strftime("%d")
    )
    directory.mkdir(parents=True, exist_ok=True)

    return directory / (
        f"FireSmoke_{area_name}_{obs_time:%Y%m%d_%H%M}.png"
    )


def list_archive_images(root, area_name):
    directory = Path(root) / area_name
    if not directory.exists():
        return []

    return sorted(
        directory.rglob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )



# ============================================================
# INTEGRATED FTP DOWNLOADER
# ============================================================

def connect_ftp(username, password, timeout=180):
    ftp = FTP(FTP_SERVER, timeout=timeout)
    ftp.login(user=username, passwd=password)
    return ftp


def safe_ftp_close(ftp):
    if ftp is None:
        return
    try:
        ftp.quit()
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass


def ahi_remote_directory(timestamp):
    return f"/jma/netcdf/{timestamp:%Y%m}/{timestamp:%d}/"


def hotspot_remote_directory(timestamp):
    return (
        f"/pub/himawari/L2/WLF/010/"
        f"{timestamp:%Y%m}/{timestamp:%d}/{timestamp:%H}/"
    )


def remote_file_size(ftp, filename):
    try:
        size = ftp.size(filename)
        return int(size) if size is not None else None
    except Exception:
        return None


def download_ftp_file(ftp, filename, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = remote_file_size(ftp, filename)

    if local_path.exists():
        if expected_size is None or local_path.stat().st_size == expected_size:
            return "skipped"

    partial_path = local_path.with_suffix(local_path.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with partial_path.open("wb") as output:
            ftp.retrbinary(
                f"RETR {filename}",
                output.write,
                blocksize=1024 * 1024,
            )

        if (
            expected_size is not None
            and partial_path.stat().st_size != expected_size
        ):
            raise IOError(
                "Ukuran file hasil download tidak cocok dengan ukuran FTP."
            )

        partial_path.replace(local_path)
        return "downloaded"

    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


def list_remote_names(ftp, remote_directory):
    ftp.cwd(remote_directory)
    return [Path(name).name for name in ftp.nlst()]


def latest_remote_ahi(ftp, now_utc, suffix, lookback_hours=6):
    """
    Cari file AHI terbaru pada hari ini/kemarin dalam jendela lookback.
    Semua menit 00,10,...,50 diterima; dashboard memakai yang paling baru.
    """
    cutoff = now_utc - timedelta(hours=lookback_hours)
    candidate_dates = {
        now_utc.date(),
        cutoff.date(),
    }

    candidates = []

    for target_date in sorted(candidate_dates):
        probe_time = datetime.combine(target_date, datetime.min.time())
        remote_dir = ahi_remote_directory(probe_time)

        try:
            names = list_remote_names(ftp, remote_dir)
        except error_perm:
            continue

        for name in names:
            timestamp = parse_ahi_time(name)
            if timestamp is None:
                continue

            if suffix and not name.endswith(f"{suffix}.nc"):
                continue

            if cutoff <= timestamp <= now_utc:
                candidates.append((timestamp, remote_dir, name))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])


def best_remote_hotspot(ftp, ahi_time, suffix, fallback_hours=2):
    """
    Cari hotspot terbaru yang tidak lebih baru dari AHI dan tidak lebih tua
    dari fallback_hours. Pencarian mencakup folder jam sebelumnya.
    """
    earliest_time = ahi_time - timedelta(hours=fallback_hours)
    cursor = earliest_time.replace(minute=0, second=0, microsecond=0)
    last_hour = ahi_time.replace(minute=0, second=0, microsecond=0)
    candidates = []

    while cursor <= last_hour:
        remote_dir = hotspot_remote_directory(cursor)

        try:
            names = list_remote_names(ftp, remote_dir)
        except error_perm:
            cursor += timedelta(hours=1)
            continue

        for name in names:
            timestamp = parse_hotspot_time(name)
            if timestamp is None:
                continue
            if suffix and not name.endswith(f"{suffix}.csv"):
                continue
            if earliest_time <= timestamp <= ahi_time:
                candidates.append((timestamp, remote_dir, name))

        cursor += timedelta(hours=1)

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])


def remove_old_netcdf(folder, retention_hours=24):
    """
    Hapus NetCDF berdasarkan timestamp pada nama file, bukan waktu modified.
    CSV hotspot tidak dihapus.
    """
    folder = Path(folder)
    if not folder.exists():
        return 0

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=retention_hours
    )

    removed = 0

    for path in folder.rglob("*.nc"):
        timestamp = parse_ahi_time(path)
        if timestamp is None:
            continue

        if timestamp < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass

    for directory in sorted(
        [p for p in folder.rglob("*") if p.is_dir()],
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    return removed


def integrated_download_latest(
    username,
    password,
    ahi_folder,
    hotspot_folder,
    ahi_suffix=DEFAULT_AHI_SUFFIX,
    hotspot_suffix=DEFAULT_HOTSPOT_SUFFIX,
    lookback_hours=6,
):
    """
    Download satu pasangan data terbaru:
    - AHI terbaru dalam lookback.
    - Hotspot persis/sedekat mungkin pada jam yang sama.
    Setelah itu hapus NetCDF yang lebih tua dari 24 jam.
    """
    if not username or not password:
        return {
            "ok": False,
            "message": "Username/password P-Tree belum tersedia.",
        }

    ftp = None

    try:
        ftp = connect_ftp(username, password)

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        ahi_remote = latest_remote_ahi(
            ftp=ftp,
            now_utc=now_utc,
            suffix=ahi_suffix,
            lookback_hours=lookback_hours,
        )

        if ahi_remote is None:
            removed = remove_old_netcdf(ahi_folder, retention_hours=24)
            return {
                "ok": False,
                "message": (
                    "Belum menemukan AHI terbaru pada jendela pemeriksaan. "
                    f"NetCDF lama yang dihapus: {removed}."
                ),
            }

        ahi_time, ahi_dir, ahi_name = ahi_remote

        ahi_local = (
            Path(ahi_folder)
            / ahi_time.strftime("%Y%m%d")
            / ahi_name
        )

        ftp.cwd(ahi_dir)
        ahi_result = download_ftp_file(
            ftp,
            ahi_name,
            ahi_local,
        )

        hotspot_remote = best_remote_hotspot(
            ftp=ftp,
            ahi_time=ahi_time,
            suffix=hotspot_suffix,
            fallback_hours=2,
        )

        hotspot_local = None
        hotspot_result = "not available"

        if hotspot_remote is not None:
            hotspot_time, hotspot_dir, hotspot_name = hotspot_remote

            hotspot_local = (
                Path(hotspot_folder)
                / hotspot_time.strftime("%Y%m%d")
                / hotspot_name
            )

            ftp.cwd(hotspot_dir)
            hotspot_result = download_ftp_file(
                ftp,
                hotspot_name,
                hotspot_local,
            )

        removed = remove_old_netcdf(
            ahi_folder,
            retention_hours=24,
        )

        return {
            "ok": True,
            "ahi_time": ahi_time,
            "ahi_path": ahi_local,
            "ahi_result": ahi_result,
            "hotspot_path": hotspot_local,
            "hotspot_result": hotspot_result,
            "removed_netcdf": removed,
            "message": (
                f"AHI {ahi_time:%Y-%m-%d %H:%M UTC}: {ahi_result}. "
                f"Hotspot: {hotspot_result}. "
                f"NetCDF >24 jam dihapus: {removed}."
            ),
        }

    except Exception as error:
        return {
            "ok": False,
            "message": f"Auto-download gagal: {error}",
        }

    finally:
        safe_ftp_close(ftp)


# ============================================================
# DATA HELPERS
# ============================================================
def find_name(dataset, candidates):
    available = set(dataset.coords) | set(dataset.variables)

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise KeyError(
        f"Tidak menemukan koordinat dari pilihan: {candidates}"
    )


def normalize(data, vmin, vmax, gamma):
    data = np.asarray(data, dtype=np.float32)

    if vmax <= vmin:
        raise ValueError("vmax harus lebih besar daripada vmin.")
    if gamma <= 0:
        raise ValueError("Gamma harus lebih besar daripada nol.")

    channel = (data - vmin) / (vmax - vmin)
    channel = np.clip(channel, 0.0, 1.0)
    return channel ** (1.0 / gamma)


def read_ahi_crop(nc_path, area):
    open_kwargs = {
        "engine": "h5netcdf",
        "decode_timedelta": False,
        "cache": False,
    }

    with xr.open_dataset(nc_path, **open_kwargs) as dataset:
        lon_name = find_name(dataset, ["longitude", "lon"])
        lat_name = find_name(dataset, ["latitude", "lat"])

        required = ["albedo_03", "albedo_04", "albedo_06"]
        missing = [band for band in required if band not in dataset]

        if missing:
            raise KeyError(f"Band tidak ditemukan: {missing}")

        longitude = dataset[lon_name]
        latitude = dataset[lat_name]

        if longitude.ndim != 1 or latitude.ndim != 1:
            raise ValueError(
                "Versi dashboard ini mengharapkan latitude dan longitude 1D."
            )

        lat_descending = float(latitude.values[0]) > float(latitude.values[-1])

        crop = dataset.sel(
            {
                lon_name: slice(area["lon_min"], area["lon_max"]),
                lat_name: (
                    slice(area["lat_max"], area["lat_min"])
                    if lat_descending
                    else slice(area["lat_min"], area["lat_max"])
                ),
            }
        )

        lon = np.asarray(crop[lon_name].values, dtype=np.float32)
        lat = np.asarray(crop[lat_name].values, dtype=np.float32)

        b03 = np.asarray(crop["albedo_03"].squeeze().values, dtype=np.float32)
        b04 = np.asarray(crop["albedo_04"].squeeze().values, dtype=np.float32)
        b06 = np.asarray(crop["albedo_06"].squeeze().values, dtype=np.float32)

    if np.nanpercentile(b03, 99.9) <= 2.0:
        b03 *= 100.0
        b04 *= 100.0
        b06 *= 100.0

    return lon, lat, b03, b04, b06


def make_smoke_rgb(b03, b04, b06, recipe):
    red = normalize(b03, recipe["r_min"], recipe["r_max"], recipe["r_gamma"])
    green = normalize(b04, recipe["g_min"], recipe["g_max"], recipe["g_gamma"])
    blue = normalize(b06, recipe["b_min"], recipe["b_max"], recipe["b_gamma"])

    rgb = np.dstack([red, green, blue])
    rgb = np.nan_to_num(rgb, nan=0.0)
    return np.clip(rgb, 0.0, 1.0)


def read_hotspot(csv_path, area, minimum_reliability):
    hotspot = pd.read_csv(csv_path, skiprows=1)
    hotspot.columns = [
        str(column).replace("# ", "").strip()
        for column in hotspot.columns
    ]

    numeric_columns = [
        "Lat",
        "Lon",
        "Area(km^2)",
        "Volcano",
        "Level",
        "Reliability",
        "FRP(Wm^-2)",
        "QF",
        "Hot(ID)",
    ]

    for column in numeric_columns:
        if column in hotspot.columns:
            hotspot[column] = pd.to_numeric(hotspot[column], errors="coerce")

    if not {"Lat", "Lon"}.issubset(hotspot.columns):
        raise KeyError("Kolom Lat dan Lon tidak ditemukan pada CSV hotspot.")

    hotspot = hotspot.dropna(subset=["Lat", "Lon"])

    if "Reliability" in hotspot.columns:
        hotspot = hotspot[hotspot["Reliability"] >= int(minimum_reliability)]

    hotspot = hotspot[
        (hotspot["Lon"] >= area["lon_min"])
        & (hotspot["Lon"] <= area["lon_max"])
        & (hotspot["Lat"] >= area["lat_min"])
        & (hotspot["Lat"] <= area["lat_max"])
    ]

    return hotspot.reset_index(drop=True)


# ============================================================
# PRODUCT HELPERS
# ============================================================
def marker_sizes(hotspot, base_size=10, marker_by_frp=False):
    sizes = np.full(len(hotspot), float(base_size))

    if (
        marker_by_frp
        and not hotspot.empty
        and "FRP(Wm^-2)" in hotspot.columns
    ):
        frp = hotspot["FRP(Wm^-2)"].fillna(0).to_numpy(dtype=float)
        if np.nanmax(frp) > np.nanmin(frp):
            sizes = float(base_size) + 70.0 * (
                (frp - np.nanmin(frp))
                / (np.nanmax(frp) - np.nanmin(frp))
            )
    return sizes


def prepare_product_data(
    nc_path,
    hotspot_path,
    area_name,
    area,
    recipe,
    minimum_reliability,
):
    observation_time = parse_ahi_time(nc_path)
    lon, lat, b03, b04, b06 = read_ahi_crop(nc_path, area)
    rgb = make_smoke_rgb(b03, b04, b06, recipe)

    hotspot = pd.DataFrame()
    if hotspot_path is not None:
        hotspot = read_hotspot(hotspot_path, area, minimum_reliability)

    total_hotspots = len(hotspot)
    high_reliability = (
        int((hotspot["Reliability"] >= 3).sum())
        if not hotspot.empty and "Reliability" in hotspot.columns
        else 0
    )

    return {
        "area_name": area_name,
        "area": area,
        "observation_time": observation_time,
        "lon": lon,
        "lat": lat,
        "rgb": rgb,
        "hotspot": hotspot,
        "total_hotspots": total_hotspots,
        "high_reliability": high_reliability,
    }


def render_png_product(product, marker_size=15, marker_by_frp=False):
    lon = product["lon"]
    lat = product["lat"]
    rgb = product["rgb"]
    hotspot = product["hotspot"]
    area = product["area"]
    area_name = product["area_name"]
    observation_time = product["observation_time"]

    figure, axis = plt.subplots(figsize=(16, 9))
    origin = "upper" if lat[0] > lat[-1] else "lower"

    axis.imshow(
        rgb,
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        origin=origin,
        interpolation="nearest",
    )

    if not hotspot.empty:
        sizes = marker_sizes(hotspot, base_size=marker_size, marker_by_frp=marker_by_frp)

        axis.scatter(
            hotspot["Lon"],
            hotspot["Lat"],
            s=sizes,
            facecolors="red",
            edgecolors="yellow",
            linewidths=0.8,
            alpha=0.93,
            zorder=20,
            label=f"Hotspot (n={len(hotspot)})",
        )
        axis.legend(loc="lower right", framealpha=0.85)

    axis.set_xlim(area["lon_min"], area["lon_max"])
    axis.set_ylim(area["lat_min"], area["lat_max"])
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.25)

    axis.set_title(
        f"{area_name} Fire & Smoke Monitor\n"
        f"{observation_time:%d %B %Y, %H:%M UTC}",
        fontweight="bold",
    )

    guide = (
        "Interpretasi cepat\n"
        "Kuning/cokelat samar: indikasi asap\n"
        "Putih terang: awan\n"
        "Titik merah-kuning: hotspot"
    )

    axis.text(
        0.012,
        0.018,
        guide,
        transform=axis.transAxes,
        color="white",
        fontsize=9,
        va="bottom",
        bbox={
            "facecolor": "black",
            "edgecolor": "white",
            "alpha": 0.68,
            "pad": 5,
        },
    )

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return buffer.getvalue()


def render_interactive_product(product, marker_size=10, marker_by_frp=False):
    lon = product["lon"]
    lat = product["lat"]
    rgb = product["rgb"]
    hotspot = product["hotspot"]
    area = product["area"]
    area_name = product["area_name"]
    observation_time = product["observation_time"]

    x0 = float(lon[0])
    dx = float((lon[-1] - lon[0]) / max(len(lon) - 1, 1))

    if lat[0] > lat[-1]:
        y0 = float(lat[0])
        dy = -abs(float((lat[-1] - lat[0]) / max(len(lat) - 1, 1)))
    else:
        y0 = float(lat[0])
        dy = abs(float((lat[-1] - lat[0]) / max(len(lat) - 1, 1)))

    figure = go.Figure()
    figure.add_trace(
        go.Image(
            z=(rgb * 255).astype(np.uint8),
            x0=x0,
            dx=dx,
            y0=y0,
            dy=dy,
            name="Smoke RGB",
        )
    )

    if not hotspot.empty:
        sizes = marker_sizes(
            hotspot,
            base_size=max(marker_size / 2, 6),
            marker_by_frp=marker_by_frp,
        )

        hover_text = []
        for _, row in hotspot.iterrows():
            lines = [
                f"Lat: {row['Lat']:.2f}",
                f"Lon: {row['Lon']:.2f}",
            ]
            for column in ["Reliability", "FRP(Wm^-2)", "Area(km^2)", "Level"]:
                if column in hotspot.columns and pd.notna(row.get(column)):
                    lines.append(f"{column}: {row[column]}")
            hover_text.append("<br>".join(lines))

        figure.add_trace(
            go.Scatter(
                x=hotspot["Lon"],
                y=hotspot["Lat"],
                mode="markers",
                marker={
                    "size": sizes,
                    "color": "red",
                    "line": {"color": "yellow", "width": 1},
                    "opacity": 0.92,
                },
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name=f"Hotspot (n={len(hotspot)})",
            )
        )

    figure.update_layout(
        title=(
            f"{area_name} Operational Fire & Smoke View<br>"
            f"<sup>{observation_time:%d %B %Y, %H:%M UTC}</sup>"
        ),
        height=760,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin={"l": 20, "r": 20, "t": 82, "b": 20},
        dragmode="zoom",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    figure.update_xaxes(
        range=[area["lon_min"], area["lon_max"]],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
    )
    figure.update_yaxes(
        range=[area["lat_min"], area["lat_max"]],
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
    )

    return figure


def data_status(age_minutes):
    if age_minutes < 60:
        return "Up to date", "success"
    if age_minutes < 120:
        return "Menunggu update Himawari", "warning"
    return "Data terlambat", "error"


def status_badge_html(status_label, status_level):
    css_class = {
        "success": "badge-green",
        "warning": "badge-yellow",
        "error": "badge-red",
    }[status_level]
    return (
        f'<span class="operational-badge {css_class}">{status_label}</span>'
    )


def kpi_card(label, value, subtext=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtext}</div>
    </div>
    """


def describe_condition(product):
    area_name = product["area_name"]
    total = product["total_hotspots"]
    high = product["high_reliability"]

    if total == 0:
        return (
            f"Tidak ada hotspot yang memenuhi filter di wilayah {area_name}. "
            f"Perlu diingat, tutupan awan dapat mengurangi kemampuan deteksi."
        )

    if high == 0:
        return (
            f"Terdeteksi {total} hotspot di {area_name}, namun belum ada titik "
            f"dengan reliability level 3."
        )

    return (
        f"Terdeteksi {total} hotspot di {area_name}; {high} di antaranya "
        f"memiliki reliability level 3. Periksa plume kuning/cokelat di sekitar "
        f"atau hilir titik hotspot."
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Operational controls")

    with st.expander("Data source", expanded=True):
        netcdf_folder = st.text_input(
            "Folder Himawari NetCDF",
            value=r"E:/move file/Project_AsapBerau/AHI",
        )
        hotspot_folder = st.text_input(
            "Folder hotspot CSV",
            value=r"E:/move file/Project_AsapBerau/HOTSPOT",
        )
        archive_folder = st.text_input(
            "Folder arsip PNG",
            value=r"E:/move file/FireSmoke_Archive",
        )

    with st.expander("JAXA auto-download", expanded=False):
        auto_download_enabled = st.checkbox(
            "Aktifkan download otomatis",
            value=False,
        )
        ptree_username = PTREE_USERNAME
        ptree_password = PTREE_PASSWORD

        if ptree_username and ptree_password:
            st.success("P-Tree credentials configured securely.")
        else:
            st.warning(
                "Credential P-Tree belum dikonfigurasi di Streamlit Secrets."
            )
        ahi_suffix = st.selectbox(
            "Grid AHI",
            ["02801_02401", "02401_02401", "07001_06001", "06001_06001"],
            index=0,
        )
        lookback_hours = st.selectbox(
            "Jendela latency AHI",
            [3, 6, 12],
            index=1,
            format_func=lambda value: f"{value} jam terakhir",
        )
        st.caption("Hotspot memakai data terbaru maksimal 2 jam sebelum waktu AHI.")
        download_now = st.button("Download terbaru", use_container_width=True)

    with st.expander("Map layers & filters", expanded=True):
        selected_area = st.radio("Area", ["Indonesia", "Berau"], horizontal=True)
        minimum_reliability = st.selectbox(
            "Hotspot confidence",
            [1, 2, 3],
            format_func=lambda value: {
                1: "Level 1–3",
                2: "Level 2–3",
                3: "Level 3",
            }[value],
            index=0,
        )
        marker_by_frp = st.checkbox("Marker mengikuti FRP", value=False)
        marker_size = st.slider("Ukuran marker", 5, 100, 10)

    refresh_now = st.button(
        "Refresh dashboard",
        type="primary",
        use_container_width=True,
    )
    st.caption("Hourly rerun aktif selama Streamlit tetap berjalan.")


# ============================================================
# AUTO REFRESH + AUTO DOWNLOAD
# ============================================================
if st_autorefresh is not None:
    st_autorefresh(
        interval=60 * 60 * 1000,
        key="hourly_refresh",
    )
else:
    st.sidebar.warning(
        "Auto-refresh hourly belum aktif karena paket "
        "`streamlit-autorefresh` belum terpasang."
    )

run_integrated_download = (
    auto_download_enabled
    or download_now
)

if run_integrated_download:
    with st.spinner("Memeriksa data terbaru di JAXA P-Tree..."):
        download_status = integrated_download_latest(
            username=ptree_username,
            password=ptree_password,
            ahi_folder=netcdf_folder,
            hotspot_folder=hotspot_folder,
            ahi_suffix=ahi_suffix,
            hotspot_suffix=DEFAULT_HOTSPOT_SUFFIX,
            lookback_hours=lookback_hours,
        )

    st.session_state.last_download_status = download_status["message"]

    if download_status["ok"]:
        st.sidebar.success(download_status["message"])
    else:
        st.sidebar.warning(download_status["message"])

if refresh_now:
    st.rerun()


# ============================================================
# MAIN TABS
# ============================================================
latest_tab, archive_tab, advanced_tab = st.tabs(
    ["Kondisi terbaru", "Arsip", "Pengaturan lanjutan"]
)


# ============================================================
# LATEST VIEW
# ============================================================
with latest_tab:
    ahi_time, latest_nc = find_latest_ahi(netcdf_folder)

    if latest_nc is None:
        st.error("Belum ada file Himawari NetCDF yang valid pada folder input.")
    else:
        hotspot_path = find_matching_hotspot(hotspot_folder, ahi_time)

        try:
            products = {}
            png_products = {}

            for area_name, area in AREA_PRESETS.items():
                products[area_name] = prepare_product_data(
                    nc_path=latest_nc,
                    hotspot_path=hotspot_path,
                    area_name=area_name,
                    area=area,
                    recipe=SMOKE_RGB,
                    minimum_reliability=minimum_reliability,
                )
                png_products[area_name] = render_png_product(
                    products[area_name],
                    marker_size=marker_size,
                    marker_by_frp=marker_by_frp,
                )

            previous_seen = st.session_state.latest_seen_ahi_time
            is_new_ahi = previous_seen is None or ahi_time > previous_seen
            if is_new_ahi:
                st.session_state.latest_seen_ahi_time = ahi_time

            for area_name in AREA_PRESETS:
                target_file = archive_path(archive_folder, area_name, ahi_time)
                product_key = f"{area_name}_{ahi_time:%Y%m%d_%H%M}_r{minimum_reliability}"

                if (
                    product_key not in st.session_state.last_processed
                    or not target_file.exists()
                ):
                    target_file.write_bytes(png_products[area_name])
                    st.session_state.last_processed[product_key] = str(target_file)

            current_product = products[selected_area]
            interactive_figure = render_interactive_product(
                current_product,
                marker_size=marker_size,
                marker_by_frp=marker_by_frp,
            )

            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            age_minutes = max(
                int((utc_now - current_product["observation_time"]).total_seconds() / 60),
                0,
            )
            status_label, status_level = data_status(age_minutes)

            st.markdown(status_badge_html(status_label, status_level), unsafe_allow_html=True)

            if is_new_ahi:
                st.success(
                    f"Data baru ditemukan: {current_product['observation_time']:%d %b %Y %H:%M UTC}."
                )
            else:
                if status_level == "success":
                    st.info(
                        "Belum ada timestamp AHI yang lebih baru. Dashboard tetap menampilkan produk terakhir."
                    )
                elif status_level == "warning":
                    st.warning(
                        "Menunggu update Himawari. Produk terakhir tetap aktif dan tidak diproses ulang."
                    )
                else:
                    st.error(
                        "Data terbaru sudah cukup lama. Periksa proses download atau ketersediaan data sumber."
                    )

            if hotspot_path is None:
                st.warning(
                    "Hotspot dalam rentang 2 jam sebelum waktu AHI belum ditemukan. Smoke RGB tetap ditampilkan tanpa overlay hotspot."
                )

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(
                    kpi_card(
                        "Waktu observasi",
                        current_product["observation_time"].strftime("%d %b %Y"),
                        current_product["observation_time"].strftime("%H:%M UTC"),
                    ),
                    unsafe_allow_html=True,
                )
            with kpi2:
                st.markdown(
                    kpi_card(
                        "Hotspot dalam area",
                        current_product["total_hotspots"],
                        f"Area aktif: {selected_area}",
                    ),
                    unsafe_allow_html=True,
                )
            with kpi3:
                st.markdown(
                    kpi_card(
                        "Reliability level 3",
                        current_product["high_reliability"],
                        "Titik confidence tertinggi",
                    ),
                    unsafe_allow_html=True,
                )
            with kpi4:
                st.markdown(
                    kpi_card(
                        "Usia data",
                        f"{age_minutes} menit",
                        f"Status: {status_label}",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
<div class="panel-card">
    <div class="panel-title">Ringkasan operasional</div>
    <div class="panel-subtitle">{describe_condition(current_product)}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            map_col, side_col = st.columns([4.5, 1.5], gap="large")

            with map_col:
                st.markdown(
                    """
<div class="panel-card">
    <div class="panel-title">Peta interaktif</div>
    <div class="panel-subtitle">Zoom, pan, dan hover untuk melihat detail hotspot.</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    interactive_figure,
                    use_container_width=True,
                    config={"displaylogo": False},
                )

                st.markdown(
                    """
<div class="legend-card">
    <div class="legend-title">Cara membaca peta</div>
    <div class="legend-item">• Kuning/cokelat samar: indikasi asap</div>
    <div class="legend-item">• Putih terang: awan tebal</div>
    <div class="legend-item">• Titik merah dengan tepi kuning: hotspot satelit</div>
    <div class="legend-item">• Bila opsi FRP aktif, marker lebih besar menandakan FRP lebih tinggi</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with side_col:
                st.markdown(
                    """
<div class="panel-card">
    <div class="panel-title">Quick actions</div>
    <div class="panel-subtitle">Unduh output operasional terbaru.</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.download_button(
                    "Download PNG Indonesia",
                    data=png_products["Indonesia"],
                    file_name=f"FireSmoke_Indonesia_{ahi_time:%Y%m%d_%H%M}.png",
                    mime="image/png",
                    use_container_width=True,
                )

                st.download_button(
                    "Download PNG Berau",
                    data=png_products["Berau"],
                    file_name=f"FireSmoke_Berau_{ahi_time:%Y%m%d_%H%M}.png",
                    mime="image/png",
                    use_container_width=True,
                )

                st.markdown(
                    """
<div class="legend-card">
    <div class="legend-title">Sumber data aktif</div>
    <div class="legend-item">AHI dan hotspot dicocokkan pada jam yang sama.</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.caption(f"AHI: `{Path(latest_nc).name}`")
                st.caption(
                    f"Hotspot: `{Path(hotspot_path).name if hotspot_path else 'belum tersedia'}`"
                )
                st.caption("Versi PNG juga otomatis disimpan ke folder arsip.")

            with st.expander("Daftar hotspot area aktif"):
                if current_product["hotspot"].empty:
                    st.info("Tidak ada hotspot dalam area aktif.")
                else:
                    preferred_columns = [
                        column
                        for column in [
                            "Lat",
                            "Lon",
                            "Reliability",
                            "FRP(Wm^-2)",
                            "Area(km^2)",
                            "Level",
                        ]
                        if column in current_product["hotspot"].columns
                    ]
                    st.dataframe(
                        current_product["hotspot"][preferred_columns],
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.exception(error)


# ============================================================
# ARCHIVE
# ============================================================
with archive_tab:
    st.markdown(
        """
<div class="panel-card">
    <div class="panel-title">Arsip PNG</div>
    <div class="panel-subtitle">Buka kembali output operasional yang sudah tersimpan.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    archive_area = st.radio(
        "Area arsip",
        ["Indonesia", "Berau"],
        horizontal=True,
        key="archive_area",
    )

    archived_images = list_archive_images(archive_folder, archive_area)

    if not archived_images:
        st.info("Belum ada PNG pada folder arsip untuk area ini.")
    else:
        labels = [
            path.relative_to(Path(archive_folder) / archive_area).as_posix()
            for path in archived_images
        ]

        selected_index = st.selectbox(
            "Pilih waktu arsip",
            options=range(len(archived_images)),
            format_func=lambda index: labels[index],
        )

        selected_image = archived_images[selected_index]
        image_bytes = selected_image.read_bytes()

        st.image(image_bytes, caption=selected_image.name, use_container_width=True)

        archive_col1, archive_col2 = st.columns([1, 2])
        with archive_col1:
            st.download_button(
                "Download PNG arsip",
                data=image_bytes,
                file_name=selected_image.name,
                mime="image/png",
                use_container_width=True,
            )
        with archive_col2:
            st.markdown(
                f"<div class='archive-note'>Lokasi file: <code>{selected_image}</code></div>",
                unsafe_allow_html=True,
            )


# ============================================================
# ADVANCED
# ============================================================
with advanced_tab:
    st.markdown(
        """
<div class="panel-card">
    <div class="panel-title">Pengaturan lanjutan</div>
    <div class="panel-subtitle">Bagian ini untuk referensi teknis. Dashboard utama sudah memakai preset operasional.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.code(
        """
Smoke RGB
R = B03, vmin 0, vmax 110, gamma 1.8
G = B04, vmin 0, vmax 100, gamma 1.0
B = B06, vmin 8, vmax 60, gamma 3.0
""".strip()
    )

    area_table = pd.DataFrame(
        [{"Area": name, **bounds} for name, bounds in AREA_PRESETS.items()]
    )

    st.markdown("#### Batas area")
    st.dataframe(area_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
#### Cara kerja otomatis

1. Aplikasi melakukan rerun otomatis setiap 1 jam bila paket `streamlit-autorefresh` tersedia.
2. Aplikasi mencari NetCDF Himawari terbaru berdasarkan timestamp nama file.
3. Bila timestamp belum berubah, dashboard tetap memakai produk terakhir.
4. Bila timestamp baru ditemukan, hotspot dicocokkan pada jam yang sama, menit `00`.
5. Dashboard membuat:
   - tampilan **interactive** untuk area yang dipilih;
   - **PNG Indonesia** untuk arsip;
   - **PNG Berau** untuk arsip.
"""
    )


st.divider()
st.caption(
    "Pemeriksaan otomatis berlangsung setiap 1 jam selama aplikasi tetap berjalan. "
    "Smoke RGB menunjukkan indikasi visual, bukan klasifikasi asap otomatis."
)

"""
Real-Time Disaster Monitoring Platform — Flask Backend
Author: Daridram | VIT University Project
Version: 2.0 — AQI, Tsunami Detection, Export, History, Rate Limiting
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
from flask import Flask, render_template, session, jsonify, request, redirect, url_for, Response
from flask_cors import CORS
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import joblib
import numpy as np
import threading
import json
import csv
import io
import os
import time
import logging


# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Logging — writes to file AND console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("disaster_platform.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Firebase Admin
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

app.secret_key = "butcher_5532c"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect('/auth')
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "332e8acd6409d7feea980d7b093495c2")
OPENWEATHER_BASE    = "https://api.openweathermap.org/data/2.5"
USGS_BASE           = "https://earthquake.usgs.gov/fdsnws/event/1/query"


# ─────────────────────────────────────────────
# IN-MEMORY RATE LIMITER
# ─────────────────────────────────────────────
rate_store = defaultdict(list)
rate_lock  = threading.Lock()

def is_rate_limited(key, limit=30, window=60):
    now = time.time()
    with rate_lock:
        rate_store[key] = [t for t in rate_store[key] if now - t < window]
        if len(rate_store[key]) >= limit:
            return True
        rate_store[key].append(now)
        return False

def rate_limit(limit=30, window=60):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ip = request.remote_addr or "unknown"
            if is_rate_limited(ip, limit, window):
                logger.warning(f"Rate limit hit: {ip} → {request.path}")
                return jsonify({"error": "Too many requests. Please wait."}), 429
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# LOAD ML MODEL
# ─────────────────────────────────────────────
try:
    model         = joblib.load("disaster_prediction_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    logger.info("ML Model loaded successfully")
except Exception as e:
    logger.warning(f"Model not found: {e}. Using mock predictions.")
    model         = None
    label_encoder = None

monitoring_cache = {}
history_cache    = {}
cache_lock       = threading.Lock()


# ─────────────────────────────────────────────
# HELPER: FETCH WEATHER
# ─────────────────────────────────────────────
def fetch_weather(lat, lon):
    try:
        url    = f"{OPENWEATHER_BASE}/weather"
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r      = requests.get(url, params=params, timeout=8)
        data   = r.json()
        return {
            "temperature": data["main"]["temp"],
            "feels_like":  data["main"]["feels_like"],
            "temp_min":    data["main"]["temp_min"],
            "temp_max":    data["main"]["temp_max"],
            "humidity":    data["main"]["humidity"],
            "pressure":    data["main"]["pressure"],
            "wind_speed":  data["wind"]["speed"] * 3.6,
            "wind_deg":    data["wind"].get("deg", 0),
            "wind_gust":   data["wind"].get("gust", 0) * 3.6,
            "rainfall":    data.get("rain", {}).get("1h", 0),
            "visibility":  round(data.get("visibility", 10000) / 1000, 1),
            "clouds":      data.get("clouds", {}).get("all", 0),
            "description": data["weather"][0]["description"].title(),
            "icon":        data["weather"][0]["icon"],
            "city":        data.get("name", "Unknown"),
            "country":     data["sys"].get("country", ""),
            "sunrise":     datetime.utcfromtimestamp(data["sys"].get("sunrise", 0)).strftime("%H:%M UTC"),
            "sunset":      datetime.utcfromtimestamp(data["sys"].get("sunset", 0)).strftime("%H:%M UTC"),
        }
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return None


# ─────────────────────────────────────────────
# HELPER: AIR QUALITY INDEX
# Uses US EPA AQI formula (0–500) computed from
# raw PM2.5 / PM10 concentrations — NOT the
# coarse 1–5 OpenWeather index which is inaccurate.
# ─────────────────────────────────────────────

def _epa_aqi(concentration, breakpoints):
    """
    Linear interpolation between EPA breakpoint pairs.
    breakpoints: list of (C_lo, C_hi, I_lo, I_hi)
    Returns integer AQI, or 500 if off-scale.
    """
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo)
    return 500

def _aqi_pm25(pm25):
    """Official EPA PM2.5 (24-hr) breakpoints."""
    bp = [
        (0.0,   12.0,   0,  50),
        (12.1,  35.4,  51, 100),
        (35.5,  55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    return _epa_aqi(round(pm25, 1), bp)

def _aqi_pm10(pm10):
    """Official EPA PM10 breakpoints."""
    bp = [
        (0,    54,   0,  50),
        (55,  154,  51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    return _epa_aqi(int(pm10), bp)

def _aqi_label_color(score):
    """Return (label, hex_color, health_tip) for a 0–500 AQI score."""
    if score <= 50:
        return ("Good",               "#00e676",
                "Air quality is satisfactory — little to no health risk.")
    elif score <= 100:
        return ("Moderate",           "#f5c518",
                "Acceptable quality. Unusually sensitive individuals may be affected.")
    elif score <= 150:
        return ("Unhealthy for SG",   "#ff6b35",
                "Sensitive groups (children, elderly, asthma) should limit outdoor time.")
    elif score <= 200:
        return ("Unhealthy",          "#ff2d55",
                "Everyone may begin to feel health effects. Reduce prolonged outdoor exertion.")
    elif score <= 300:
        return ("Very Unhealthy",     "#a78bfa",
                "Health alert — serious effects for everyone. Avoid outdoor activity.")
    else:
        return ("Hazardous",          "#8b0000",
                "Emergency conditions. Everyone at risk. Stay indoors with windows closed.")

def fetch_aqi(lat, lon):
    try:
        url    = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
        r      = requests.get(url, params=params, timeout=8)
        data   = r.json()
        comp   = data["list"][0]["components"]

        # Raw concentrations (μg/m³)
        pm2_5 = float(comp.get("pm2_5", 0) or 0)
        pm10  = float(comp.get("pm10",  0) or 0)
        no2   = float(comp.get("no2",   0) or 0)
        o3    = float(comp.get("o3",    0) or 0)
        so2   = float(comp.get("so2",   0) or 0)
        co_ug = float(comp.get("co",    0) or 0)   # μg/m³ from API

        # Compute individual sub-indices
        sub_pm25 = _aqi_pm25(pm2_5)
        sub_pm10 = _aqi_pm10(pm10)

        # Overall AQI = highest sub-index (EPA standard)
        aqi_score = max(sub_pm25, sub_pm10)
        dominant  = "PM2.5" if sub_pm25 >= sub_pm10 else "PM10"

        label, color, tip = _aqi_label_color(aqi_score)

        return {
            "aqi":      aqi_score,          # US EPA 0–500 integer
            "sub_pm25": sub_pm25,
            "sub_pm10": sub_pm10,
            "dominant": dominant,
            "label":    label,
            "color":    color,
            "tip":      tip,
            "pm2_5":    round(pm2_5, 1),
            "pm10":     round(pm10,  1),
            "no2":      round(no2,   1),
            "o3":       round(o3,    1),
            "so2":      round(so2,   1),
            "co":       round(co_ug / 1000, 2),   # convert μg/m³ → mg/m³
        }
    except Exception as e:
        logger.error(f"AQI error: {e}")
        return None


# ─────────────────────────────────────────────
# HELPER: FETCH FORECAST
# ─────────────────────────────────────────────
def fetch_forecast(lat, lon):
    try:
        url    = f"{OPENWEATHER_BASE}/forecast"
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric", "cnt": 40}
        r      = requests.get(url, params=params, timeout=8)
        data   = r.json()
        forecasts = []
        for item in data.get("list", []):
            forecasts.append({
                "timestamp":   item["dt_txt"],
                "rainfall":    item.get("rain", {}).get("3h", 0),
                "wind_speed":  item["wind"]["speed"] * 3.6,
                "wind_gust":   item["wind"].get("gust", 0) * 3.6,
                "humidity":    item["main"]["humidity"],
                "temperature": item["main"]["temp"],
                "feels_like":  item["main"]["feels_like"],
                "pressure":    item["main"]["pressure"],
                "description": item["weather"][0]["description"].title(),
                "icon":        item["weather"][0]["icon"],
                "pop":         round(item.get("pop", 0) * 100),
            })
        return forecasts
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return []


# ─────────────────────────────────────────────
# HELPER: FETCH EARTHQUAKES
# ─────────────────────────────────────────────
def fetch_earthquakes(lat, lon, radius_km=500):
    try:
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        params = {
            "format":       "geojson",
            "starttime":    start_time.strftime("%Y-%m-%d"),
            "endtime":      end_time.strftime("%Y-%m-%d"),
            "latitude":     lat, "longitude": lon,
            "maxradiuskm":  radius_km,
            "minmagnitude": 2.0,
            "orderby":      "magnitude",
            "limit":        10,
        }
        r    = requests.get(USGS_BASE, params=params, timeout=8)
        data = r.json()
        quakes = []
        for feature in data.get("features", []):
            p = feature["properties"]
            c = feature["geometry"]["coordinates"]
            quakes.append({
                "magnitude": p.get("mag", 0),
                "place":     p.get("place", "Unknown"),
                "time":      datetime.utcfromtimestamp(p["time"] / 1000).strftime("%Y-%m-%d %H:%M UTC"),
                "depth_km":  c[2],
                "lon": c[0], "lat": c[1],
                "url":       p.get("url", ""),
                "type":      p.get("type", "earthquake"),
                "felt":      p.get("felt", 0) or 0,
                "alert":     p.get("alert"),
            })
        return quakes
    except Exception as e:
        logger.error(f"USGS error: {e}")
        return []


# ─────────────────────────────────────────────
# HELPER: TSUNAMI RISK DETECTION
# ─────────────────────────────────────────────
def detect_tsunami_risk(quakes, lat, lon):
    risk = {"level": "NONE", "reason": "", "quakes": []}
    for q in quakes:
        mag   = q.get("magnitude", 0)
        depth = q.get("depth_km", 999)
        if mag >= 7.0 and depth <= 70:
            risk["level"]  = "HIGH" if mag >= 8.0 else "MEDIUM"
            risk["reason"] = f"M{mag} shallow quake ({depth}km deep) — potential tsunami generation"
            risk["quakes"].append(q)
        elif mag >= 6.5 and depth <= 35 and risk["level"] == "NONE":
            risk["level"]  = "LOW"
            risk["reason"] = f"M{mag} very shallow quake — monitor for coastal impact"
            risk["quakes"].append(q)
    return risk


# ─────────────────────────────────────────────
# HELPER: ML PREDICTION
# ─────────────────────────────────────────────
def predict_disaster(lat, lon, rainfall, magnitude, wind_kmh):
    features = np.array([[lat, lon, rainfall, magnitude, wind_kmh]])
    if model is None:
        if rainfall  > 100: return "Flood",      0.85
        if magnitude > 5.0: return "Earthquake", 0.90
        if wind_kmh  > 90:  return "Cyclone",    0.78
        return "No Disaster", 0.95
    try:
        proba = model.predict_proba(features)[0]
        idx   = np.argmax(proba)
        label = label_encoder.inverse_transform([idx])[0]
        return label, float(proba[idx])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Unknown", 0.0


# ─────────────────────────────────────────────
# HELPER: RISK LEVEL
# ─────────────────────────────────────────────
def compute_risk(disaster_type, confidence, weather, quakes):
    level = "LOW"
    if disaster_type in ("Flood", "Cyclone") and confidence > 0.7:
        level = "HIGH" if confidence > 0.85 else "MEDIUM"
    if disaster_type == "Earthquake" and quakes:
        max_mag = max(q["magnitude"] for q in quakes)
        if max_mag >= 6.0:   level = "CRITICAL"
        elif max_mag >= 4.5: level = "HIGH"
        elif max_mag >= 3.0: level = "MEDIUM"
    if weather and weather.get("wind_speed", 0) > 120:
        level = "CRITICAL"
    return level


# ─────────────────────────────────────────────
# BACKGROUND JOB
# ─────────────────────────────────────────────
def refresh_location(lat, lon):
    key     = f"{lat},{lon}"
    weather = fetch_weather(lat, lon)
    if not weather:
        return

    forecast  = fetch_forecast(lat, lon)
    quakes    = fetch_earthquakes(lat, lon)
    aqi       = fetch_aqi(lat, lon)
    magnitude = quakes[0]["magnitude"] if quakes else 0.0
    disaster, confidence = predict_disaster(lat, lon, weather["rainfall"], magnitude, weather["wind_speed"])

    future_risks = []
    for fc in forecast[:8]:
        fd, fc_conf = predict_disaster(lat, lon, fc["rainfall"], 0, fc["wind_speed"])
        if fd != "No Disaster":
            future_risks.append({
                "time":       fc["timestamp"],
                "disaster":   fd,
                "confidence": round(fc_conf, 2),
                "wind_speed": round(fc["wind_speed"], 1),
                "rainfall":   round(fc["rainfall"], 1),
            })

    risk_level   = compute_risk(disaster, confidence, weather, quakes)
    tsunami_risk = detect_tsunami_risk(quakes, lat, lon)

    entry = {
        "updated_at":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "location":     {"lat": lat, "lon": lon, "city": weather.get("city", ""), "country": weather.get("country", "")},
        "weather":      weather,
        "aqi":          aqi,
        "current":      {"disaster": disaster, "confidence": round(confidence, 2), "risk_level": risk_level},
        "earthquakes":  quakes,
        "tsunami_risk": tsunami_risk,
        "future_risks": future_risks,
        "forecast_raw": forecast[:40],
    }

    with cache_lock:
        monitoring_cache[key] = entry
        if key not in history_cache:
            history_cache[key] = []
        history_cache[key].append({
            "time":        entry["updated_at"],
            "disaster":    disaster,
            "risk_level":  risk_level,
            "confidence":  round(confidence, 2),
            "temperature": weather.get("temperature", 0),
            "rainfall":    weather.get("rainfall", 0),
            "wind_speed":  weather.get("wind_speed", 0),
            "aqi":         aqi["aqi"] if aqi else None,
        })
        if len(history_cache[key]) > 48:
            history_cache[key] = history_cache[key][-48:]

    logger.info(f"Refreshed {key}: {disaster} ({risk_level})")


def scheduled_monitor():
    with cache_lock:
        keys = list(monitoring_cache.keys())
    for key in keys:
        lat, lon = map(float, key.split(","))
        refresh_location(lat, lon)


def send_hourly_alerts():
    logger.info("Hourly alert check running...")


scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_monitor,  "interval", minutes=5, id="monitor")
scheduler.add_job(send_hourly_alerts, "interval", hours=1,   id="alerts")
scheduler.start()
logger.info("Background scheduler started")


# ─────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    if 'user' in session: return redirect('/dashboard')
    return redirect('/auth')

@app.route("/auth")
def auth_page():
    return render_template('auth.html')

@app.route("/register")
def register_page():
    return render_template('register.html')

@app.route("/forgot-password")
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route("/dashboard")
def dashboard():
    return render_template('index.html')


# ─────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────
@app.route("/api/auth/verify", methods=["POST"])
def verify_token():
    data  = request.get_json()
    token = data.get('token')
    try:
        decoded = firebase_auth.verify_id_token(token)
        session['user'] = {
            'uid':           decoded['uid'],
            'email':         data.get('emailOrPhone'),
            'name':          data.get('name') or decoded.get('name', 'User'),
            'notifications': data.get('notifications_enabled', False)
        }
        return jsonify({'success': True, 'user': session['user']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

@app.route("/api/auth/logout")
def logout():
    session.clear()
    return redirect('/auth')


# ─────────────────────────────────────────────
# ROUTES — CORE MONITORING
# ─────────────────────────────────────────────
@app.route("/api/monitor", methods=["POST"])
@rate_limit(limit=30, window=60)
def start_monitoring():
    data = request.get_json()
    lat  = float(data.get("lat", 0))
    lon  = float(data.get("lon", 0))
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"status": "error", "message": "Invalid coordinates"}), 400
    key = f"{lat},{lon}"
    with cache_lock:
        if key not in monitoring_cache:
            monitoring_cache[key] = {}
    refresh_location(lat, lon)
    with cache_lock:
        result = monitoring_cache.get(key, {})
    return jsonify({"status": "ok", "data": result})


@app.route("/api/status", methods=["GET"])
def get_status():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    key = f"{lat},{lon}"
    with cache_lock:
        result = monitoring_cache.get(key)
    if not result:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": "ok", "data": result})


@app.route("/api/predict", methods=["POST"])
@rate_limit(limit=20, window=60)
def manual_predict():
    data      = request.get_json()
    lat       = float(data.get("lat", 0))
    lon       = float(data.get("lon", 0))
    rainfall  = float(data.get("rainfall", 0))
    magnitude = float(data.get("magnitude", 0))
    wind_kmh  = float(data.get("wind_kmh", 0))
    disaster, confidence = predict_disaster(lat, lon, rainfall, magnitude, wind_kmh)
    return jsonify({"prediction": disaster, "confidence": round(confidence, 2)})


@app.route("/api/search", methods=["GET"])
@rate_limit(limit=20, window=60)
def search_city():
    city = request.args.get("q", "")
    try:
        url    = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city, "limit": 5, "appid": OPENWEATHER_API_KEY}
        r      = requests.get(url, params=params, timeout=6)
        cities = [{"name": c["name"], "country": c.get("country",""), "lat": c["lat"], "lon": c["lon"]}
                  for c in r.json()]
        return jsonify({"results": cities})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)})


@app.route("/api/earthquakes/global", methods=["GET"])
def global_earthquakes():
    try:
        params = {
            "format": "geojson",
            "starttime": (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "minmagnitude": 4.5, "orderby": "magnitude", "limit": 50,
        }
        r    = requests.get(USGS_BASE, params=params, timeout=8)
        data = r.json()
        quakes = []
        for f in data.get("features", []):
            p = f["properties"]
            c = f["geometry"]["coordinates"]
            quakes.append({
                "lat": c[1], "lon": c[0],
                "magnitude": p.get("mag", 0),
                "place":     p.get("place", ""),
                "depth":     c[2],
                "time":      datetime.utcfromtimestamp(p["time"]/1000).strftime("%Y-%m-%d %H:%M UTC"),
                "felt":      p.get("felt", 0) or 0,
                "alert":     p.get("alert"),
            })
        return jsonify({"quakes": quakes})
    except Exception as e:
        return jsonify({"quakes": [], "error": str(e)})


# ─────────────────────────────────────────────
# ROUTES — NEW FEATURES
# ─────────────────────────────────────────────

@app.route("/api/aqi", methods=["GET"])
@rate_limit(limit=20, window=60)
def get_aqi():
    """Real-time Air Quality Index for a location."""
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        aqi = fetch_aqi(lat, lon)
        if aqi:
            return jsonify({"status": "ok", "aqi": aqi})
        return jsonify({"status": "error", "message": "AQI unavailable"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/tsunami", methods=["GET"])
@rate_limit(limit=15, window=60)
def get_tsunami_risk():
    """Tsunami risk evaluation based on nearby seismic activity."""
    try:
        lat    = float(request.args.get("lat", 0))
        lon    = float(request.args.get("lon", 0))
        quakes = fetch_earthquakes(lat, lon, radius_km=1000)
        risk   = detect_tsunami_risk(quakes, lat, lon)
        return jsonify({"status": "ok", "tsunami_risk": risk, "quakes_checked": len(quakes)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/history", methods=["GET"])
def get_history():
    """Last 48 monitoring snapshots for a location."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    key = f"{lat},{lon}"
    with cache_lock:
        history = history_cache.get(key, [])
    return jsonify({"status": "ok", "history": history, "count": len(history)})


@app.route("/api/export", methods=["GET"])
def export_data():
    """Export monitoring data as JSON or CSV. Params: lat, lon, format."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    fmt = request.args.get("format", "json").lower()
    key = f"{lat},{lon}"

    with cache_lock:
        data    = monitoring_cache.get(key)
        history = history_cache.get(key, [])

    if not data:
        return jsonify({"error": "No data found. Monitor this location first."}), 404

    loc  = data.get("location", {})
    slug = loc.get("city", "location").replace(" ", "_").lower()
    ts   = datetime.utcnow().strftime('%Y%m%d_%H%M')

    if fmt == "csv":
        out    = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["Timestamp", "City", "Country", "Disaster", "Risk Level", "Confidence %",
                         "Temperature °C", "Humidity %", "Rainfall mm/h", "Wind km/h",
                         "Pressure hPa", "AQI", "Feels Like °C", "Visibility km"])
        w   = data.get("weather", {})
        c   = data.get("current", {})
        aqi = data.get("aqi") or {}
        writer.writerow([
            data.get("updated_at",""), loc.get("city",""), loc.get("country",""),
            c.get("disaster",""), c.get("risk_level",""), round((c.get("confidence",0))*100,1),
            w.get("temperature",""), w.get("humidity",""), w.get("rainfall",""),
            w.get("wind_speed",""), w.get("pressure",""), aqi.get("aqi",""),
            w.get("feels_like",""), w.get("visibility","")
        ])
        if history:
            writer.writerow([])
            writer.writerow(["── HISTORY ──"])
            writer.writerow(["Time","Disaster","Risk Level","Confidence %","Temp °C","Rainfall mm/h","Wind km/h","AQI"])
            for h in history:
                writer.writerow([h.get("time",""), h.get("disaster",""), h.get("risk_level",""),
                                  round((h.get("confidence",0))*100,1),
                                  h.get("temperature",""), h.get("rainfall",""),
                                  h.get("wind_speed",""), h.get("aqi","")])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=disastersense_{slug}_{ts}.csv"})
    else:
        payload = {"exported_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "current": data, "history": history}
        return Response(json.dumps(payload, indent=2), mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename=disastersense_{slug}_{ts}.json"})


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """All currently monitored locations ranked by risk."""
    with cache_lock:
        all_data = dict(monitoring_cache)
    if not all_data:
        return jsonify({"status": "ok", "summary": [], "total": 0})
    rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
    summary = []
    for key, d in all_data.items():
        if not d: continue
        loc = d.get("location", {})
        cur = d.get("current", {})
        summary.append({
            "city": loc.get("city", key), "country": loc.get("country",""),
            "lat": loc.get("lat"), "lon": loc.get("lon"),
            "disaster": cur.get("disaster","Unknown"),
            "risk_level": cur.get("risk_level","UNKNOWN"),
            "confidence": cur.get("confidence", 0),
            "updated_at": d.get("updated_at",""),
        })
    summary.sort(key=lambda x: rank.get(x["risk_level"], 0), reverse=True)
    return jsonify({"status": "ok", "summary": summary, "total": len(summary)})


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":            "running",
        "version":           "2.0",
        "model_loaded":      model is not None,
        "cached_locations":  len(monitoring_cache),
        "history_entries":   sum(len(v) for v in history_cache.values()),
        "scheduler_running": scheduler.running,
        "timestamp":         datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    })


# ─────────────────────────────────────────────
# TEMPLATE CONTEXT + ERROR HANDLERS
# ─────────────────────────────────────────────
@app.context_processor
def inject_user():
    return dict(current_user=session.get('user'))

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def too_many(e):
    return jsonify({"error": "Rate limit exceeded"}), 429


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True, use_reloader=False)
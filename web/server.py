from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.database import (
    get_candles,
    get_health,
    get_latest_ai_analysis,
    get_latest_candle,
    get_portfolio_summary,
    get_positions,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="AITrader Dashboard",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def api_health():
    return get_health()


@app.get("/api/market/latest")
def market_latest():
    return get_latest_candle()


@app.get("/api/market/candles")
def market_candles(
    timeframe: str = Query("1m", pattern="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(200, ge=1, le=1000),
):
    return get_candles(timeframe=timeframe, limit=limit)


@app.get("/api/portfolio/positions")
def portfolio_positions():
    return get_positions()


@app.get("/api/portfolio/summary")
def portfolio_summary():
    return get_portfolio_summary()



@app.get("/api/ai/latest")
def ai_latest():
    analysis = get_latest_ai_analysis()

    if analysis is None:
        return {
            "status": "no_data",
            "data": None,
        }

    return {
        "status": "ok",
        "data": analysis,
    }

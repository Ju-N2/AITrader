from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from database import (
    get_health,
    get_latest_candle,
    get_candles,
    get_positions,
    get_portfolio_summary,
    get_latest_ai_analysis,
)

app = FastAPI(
    title="AITrader Dashboard",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory="web/static"),
    name="static",
)


@app.get("/")
def dashboard():
    return FileResponse("web/static/index.html")


@app.get("/api/health")
def api_health():
    try:
        return get_health()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/market/latest")
def market_latest():
    try:
        return get_latest_candle()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/market/candles")
def market_candles(
    timeframe: str = Query("1m", pattern="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(200, ge=1, le=1000),
):
    try:
        return get_candles(timeframe=timeframe, limit=limit)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/portfolio/positions")
def portfolio_positions():
    try:
        return get_positions()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        return get_portfolio_summary()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))



@app.get("/api/ai/latest")
def ai_latest():
    try:
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

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

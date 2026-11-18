const $ = (id) => document.getElementById(id);

function formatPrice(value) {
    return Number(value).toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatTime(timestamp) {
    return new Date(Number(timestamp)).toLocaleString("ko-KR");
}

let selectedTimeframe = "1m";
let marketRequestId = 0;

const TIMEFRAME_LABELS = {
    "1m": "1-minute",
    "5m": "5-minute",
    "15m": "15-minute",
    "1h": "1-hour",
    "4h": "4-hour"
};

async function loadMarket() {
    const requestId = ++marketRequestId;

    const response = await fetch(
        `/api/market/candles?timeframe=${selectedTimeframe}&limit=150`,
        { cache: "no-store" }
    );

    if (!response.ok) {
        throw new Error("Market candles API request failed");
    }

    const candles = await response.json();

    // Ignore an older response if the selected timeframe changed.
    if (requestId !== marketRequestId) return;

    const latest = candles.length
        ? candles[candles.length - 1]
        : null;

    if (latest) {
        $("current-price").textContent =
            formatPrice(latest.close);

        $("candle-time").textContent =
            formatTime(latest.timestamp_ms);
    } else {
        $("current-price").textContent = "--";
        $("candle-time").textContent = "No candle data";
    }

    const label = TIMEFRAME_LABELS[selectedTimeframe];

    $("chart-subtitle").textContent =
        `BTC/USDT · ${label} candlesticks`;

    $("current-timeframe-label").textContent =
        `${selectedTimeframe.toUpperCase()} · OHLC + VOLUME`;

    $("chart-count").textContent =
        `${candles.length} candles`;

    drawChart(candles);
}

async function loadPortfolio() {
    const [summaryResponse, positionsResponse] = await Promise.all([
        fetch("/api/portfolio/summary"),
        fetch("/api/portfolio/positions")
    ]);

    if (!summaryResponse.ok || !positionsResponse.ok) {
        throw new Error("Portfolio API request failed");
    }

    const summary = await summaryResponse.json();
    const positions = await positionsResponse.json();

    $("open-positions").textContent = summary.open_positions;
    $("realized-pnl").textContent =
        formatPrice(summary.realized_pnl);

    const body = $("positions-body");

    if (!positions.length) {
        body.innerHTML =
            '<tr><td colspan="6" class="empty">No open positions</td></tr>';
        return;
    }

    body.innerHTML = positions.map(position => `
        <tr>
            <td>${position.symbol}</td>
            <td>${position.side}</td>
            <td>${formatPrice(position.entry_price)}</td>
            <td>${position.remaining_quantity}</td>
            <td>${formatPrice(position.stop_price)}</td>
            <td>${position.status}</td>
        </tr>
    `).join("");
}

async function loadHealth() {
    const response = await fetch("/api/health");

    if (!response.ok) {
        throw new Error("Health API request failed");
    }

    const health = await response.json();

    $("connection-dot").style.background = "var(--green)";
    $("connection-text").textContent = "Connected";
    $("api-status").textContent = "Online";
    $("db-status").textContent = health.status;
}


const indicatorVisibility = {
    ema12: true,
    ema26: true,
    sma20: false,
    sma50: false
};

let chartCandles = [];
let chartIndicators = {};
let chartHoverIndex = null;

function setupCanvas(canvasId) {
    const canvas = $(canvasId);
    const rect = canvas.getBoundingClientRect();

    if (rect.width <= 0 || rect.height <= 0) {
        return null;
    }

    const ratio = Math.max(1, window.devicePixelRatio || 1);

    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    return {
        ctx,
        width: rect.width,
        height: rect.height
    };
}

function calculateEMA(values, period) {
    const result = Array(values.length).fill(null);

    let buffer = [];
    let ema = null;
    const multiplier = 2 / (period + 1);

    for (let i = 0; i < values.length; i++) {
        const value = values[i];

        if (value === null || !Number.isFinite(value)) {
            continue;
        }

        if (ema === null) {
            buffer.push(value);

            if (buffer.length === period) {
                ema = buffer.reduce((sum, item) => sum + item, 0) / period;
                result[i] = ema;
            }
        } else {
            ema = (value - ema) * multiplier + ema;
            result[i] = ema;
        }
    }

    return result;
}

function calculateSMA(values, period) {
    const result = Array(values.length).fill(null);

    for (let i = period - 1; i < values.length; i++) {
        let sum = 0;
        let valid = true;

        for (let j = i - period + 1; j <= i; j++) {
            if (!Number.isFinite(values[j])) {
                valid = false;
                break;
            }

            sum += values[j];
        }

        if (valid) {
            result[i] = sum / period;
        }
    }

    return result;
}

function calculateRSI(values, period = 14) {
    const result = Array(values.length).fill(null);

    if (values.length <= period) {
        return result;
    }

    let averageGain = 0;
    let averageLoss = 0;

    for (let i = 1; i <= period; i++) {
        const difference = values[i] - values[i - 1];

        averageGain += Math.max(difference, 0);
        averageLoss += Math.max(-difference, 0);
    }

    averageGain /= period;
    averageLoss /= period;

    function getRSI(gain, loss) {
        if (loss === 0) return gain === 0 ? 50 : 100;

        const rs = gain / loss;
        return 100 - (100 / (1 + rs));
    }

    result[period] = getRSI(averageGain, averageLoss);

    for (let i = period + 1; i < values.length; i++) {
        const difference = values[i] - values[i - 1];

        const gain = Math.max(difference, 0);
        const loss = Math.max(-difference, 0);

        averageGain =
            ((averageGain * (period - 1)) + gain) / period;

        averageLoss =
            ((averageLoss * (period - 1)) + loss) / period;

        result[i] = getRSI(averageGain, averageLoss);
    }

    return result;
}

function calculateMACD(values) {
    const ema12 = calculateEMA(values, 12);
    const ema26 = calculateEMA(values, 26);

    const macd = values.map((_, index) => {
        if (ema12[index] === null || ema26[index] === null) {
            return null;
        }

        return ema12[index] - ema26[index];
    });

    const signal = calculateEMA(macd, 9);

    const histogram = macd.map((value, index) => {
        if (value === null || signal[index] === null) {
            return null;
        }

        return value - signal[index];
    });

    return { macd, signal, histogram };
}

function calculateIndicators(candles) {
    const closes = candles.map(candle => candle.close);

    return {
        ema12: calculateEMA(closes, 12),
        ema26: calculateEMA(closes, 26),
        sma20: calculateSMA(closes, 20),
        sma50: calculateSMA(closes, 50),
        rsi: calculateRSI(closes, 14),
        macd: calculateMACD(closes)
    };
}

function drawSeries(ctx, values, xFor, yFor, color, lineWidth = 1.5) {
    ctx.beginPath();

    let started = false;

    values.forEach((value, index) => {
        if (!Number.isFinite(value)) {
            started = false;
            return;
        }

        const x = xFor(index);
        const y = yFor(value);

        if (!started) {
            ctx.moveTo(x, y);
            started = true;
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
}

function drawChart(candles) {
    chartCandles = candles
        .map(candle => ({
            timestamp: Number(candle.timestamp_ms),
            open: Number(candle.open),
            high: Number(candle.high),
            low: Number(candle.low),
            close: Number(candle.close),
            volume: Math.max(0, Number(candle.volume) || 0)
        }))
        .filter(candle =>
            Number.isFinite(candle.open) &&
            Number.isFinite(candle.high) &&
            Number.isFinite(candle.low) &&
            Number.isFinite(candle.close) &&
            Number.isFinite(candle.timestamp) &&
            candle.high >= candle.low
        );

    chartHoverIndex = null;
    $("chart-tooltip").hidden = true;

    chartIndicators = calculateIndicators(chartCandles);

    renderTechnicalCharts();
}

function renderTechnicalCharts() {
    drawPriceChart();
    drawRSIChart();
    drawMACDChart();
}

function drawPriceChart() {
    const surface = setupCanvas("price-chart");

    if (!surface) return;

    const { ctx, width, height } = surface;
    const candles = chartCandles;

    if (!candles.length) {
        ctx.fillStyle = "#8491a7";
        ctx.font = "12px sans-serif";
        ctx.fillText("No candle data available", 18, 28);
        return;
    }

    const left = 80;
    const right = 16;
    const top = 18;
    const bottom = 26;

    const priceHeight = height * 0.66;
    const volumeTop = top + priceHeight + 32;
    const volumeHeight = Math.max(25, height - volumeTop - bottom);
    const plotWidth = width - left - right;

    const highest = Math.max(...candles.map(c => c.high));
    const lowest = Math.min(...candles.map(c => c.low));
    const range = highest - lowest || Math.abs(highest) * 0.001 || 1;

    const upper = highest + range * 0.055;
    const lower = lowest - range * 0.055;

    const yFor = value =>
        top + ((upper - value) / (upper - lower)) * priceHeight;

    const step = plotWidth / candles.length;
    const xFor = index => left + (index + 0.5) * step;
    const bodyWidth = Math.max(1.5, Math.min(12, step * 0.66));

    // Price grid.
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= 4; i++) {
        const fraction = i / 4;
        const y = top + priceHeight * fraction;
        const price = upper - (upper - lower) * fraction;

        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);

        ctx.strokeStyle = "rgba(145,160,185,0.13)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "#8491a7";
        ctx.fillText(formatPrice(price), left - 10, y);
    }

    // Volume bars.
    const maxVolume = Math.max(
        ...candles.map(candle => candle.volume),
        1
    );

    ctx.beginPath();
    ctx.moveTo(left, volumeTop - 10);
    ctx.lineTo(width - right, volumeTop - 10);
    ctx.strokeStyle = "rgba(145,160,185,0.16)";
    ctx.stroke();

    candles.forEach((candle, index) => {
        const x = xFor(index);
        const bullish = candle.close >= candle.open;
        const color = bullish ? "#42e6ad" : "#ff6d83";

        const openY = yFor(candle.open);
        const closeY = yFor(candle.close);

        // Wick.
        ctx.beginPath();
        ctx.moveTo(x, yFor(candle.high));
        ctx.lineTo(x, yFor(candle.low));
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Body.
        ctx.fillStyle = color;
        ctx.fillRect(
            x - bodyWidth / 2,
            Math.min(openY, closeY),
            bodyWidth,
            Math.max(1.5, Math.abs(closeY - openY))
        );

        // Volume.
        const volumeBarHeight =
            (candle.volume / maxVolume) * volumeHeight;

        ctx.globalAlpha = 0.43;
        ctx.fillRect(
            x - bodyWidth / 2,
            height - bottom - volumeBarHeight,
            bodyWidth,
            Math.max(1, volumeBarHeight)
        );
        ctx.globalAlpha = 1;
    });

    // Moving averages.
    const overlays = [
        ["ema12", "#ffe08a", 1.7],
        ["ema26", "#9d9fff", 1.7],
        ["sma20", "#53c7ed", 1.5],
        ["sma50", "#e6a0ff", 1.5]
    ];

    overlays.forEach(([key, color, lineWidth]) => {
        if (indicatorVisibility[key]) {
            drawSeries(
                ctx,
                chartIndicators[key],
                xFor,
                yFor,
                color,
                lineWidth
            );
        }
    });

    // Last close reference line.
    const last = candles[candles.length - 1];
    const lastY = yFor(last.close);

    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(left, lastY);
    ctx.lineTo(width - right, lastY);
    ctx.strokeStyle = "rgba(141,158,255,0.72)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // Time labels.
    const indexes = [
        0,
        Math.floor((candles.length - 1) / 2),
        candles.length - 1
    ];

    ctx.fillStyle = "#8491a7";
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "top";

    indexes.forEach((index, position) => {
        const label = new Date(
            candles[index].timestamp
        ).toLocaleString("ko-KR", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit"
        });

        ctx.textAlign =
            position === 0 ? "left" :
            position === 2 ? "right" : "center";

        ctx.fillText(label, xFor(index), height - bottom + 9);
    });

    // Hover crosshair.
    if (
        chartHoverIndex !== null &&
        chartHoverIndex >= 0 &&
        chartHoverIndex < candles.length
    ) {
        const x = xFor(chartHoverIndex);

        ctx.beginPath();
        ctx.setLineDash([3, 4]);
        ctx.moveTo(x, top);
        ctx.lineTo(x, height - bottom);
        ctx.strokeStyle = "rgba(235,240,255,0.5)";
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function drawRSIChart() {
    const surface = setupCanvas("rsi-chart");

    if (!surface) return;

    const { ctx, width, height } = surface;
    const values = chartIndicators.rsi || [];

    const left = 80;
    const right = 16;
    const top = 10;
    const bottom = 14;

    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const step = plotWidth / Math.max(values.length, 1);

    const xFor = index => left + (index + 0.5) * step;
    const yFor = value =>
        top + ((100 - value) / 100) * plotHeight;

    [70, 50, 30].forEach(level => {
        const y = yFor(level);

        ctx.beginPath();
        ctx.setLineDash(level === 50 ? [2, 4] : [5, 4]);
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);

        ctx.strokeStyle = level === 50
            ? "rgba(145,160,185,0.12)"
            : "rgba(255,109,131,0.35)";

        ctx.stroke();
    });

    ctx.setLineDash([]);

    ctx.fillStyle = "#8491a7";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    [70, 50, 30].forEach(level => {
        ctx.fillText(String(level), left - 10, yFor(level));
    });

    drawSeries(
        ctx,
        values,
        xFor,
        yFor,
        "#b59aff",
        1.8
    );

    if (chartHoverIndex !== null && values[chartHoverIndex] !== null) {
        const x = xFor(chartHoverIndex);

        ctx.beginPath();
        ctx.setLineDash([3, 4]);
        ctx.moveTo(x, top);
        ctx.lineTo(x, height - bottom);
        ctx.strokeStyle = "rgba(235,240,255,0.35)";
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function drawMACDChart() {
    const surface = setupCanvas("macd-chart");

    if (!surface) return;

    const { ctx, width, height } = surface;
    const { macd, signal, histogram } = chartIndicators.macd;

    const left = 80;
    const right = 16;
    const top = 12;
    const bottom = 12;

    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;

    const allValues = [...macd, ...signal, ...histogram]
        .filter(Number.isFinite);

    const maxAbs = Math.max(
        ...allValues.map(Math.abs),
        0.000001
    ) * 1.12;

    const step = plotWidth / Math.max(macd.length, 1);
    const xFor = index => left + (index + 0.5) * step;
    const centerY = top + plotHeight / 2;
    const yFor = value =>
        centerY - (value / maxAbs) * (plotHeight / 2);

    // Zero line.
    ctx.beginPath();
    ctx.moveTo(left, centerY);
    ctx.lineTo(width - right, centerY);
    ctx.strokeStyle = "rgba(145,160,185,0.3)";
    ctx.stroke();

    const barWidth = Math.max(1.5, Math.min(10, step * 0.65));

    histogram.forEach((value, index) => {
        if (!Number.isFinite(value)) return;

        const x = xFor(index);
        const y = yFor(value);

        ctx.fillStyle = value >= 0 ? "#42e6ad" : "#ff6d83";

        ctx.globalAlpha = 0.65;

        ctx.fillRect(
            x - barWidth / 2,
            Math.min(y, centerY),
            barWidth,
            Math.max(1, Math.abs(centerY - y))
        );

        ctx.globalAlpha = 1;
    });

    drawSeries(
        ctx,
        macd,
        xFor,
        yFor,
        "#8d9eff",
        1.7
    );

    drawSeries(
        ctx,
        signal,
        xFor,
        yFor,
        "#ffcf81",
        1.5
    );

    ctx.fillStyle = "#8491a7";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("0", left - 10, centerY);

    if (chartHoverIndex !== null) {
        const x = xFor(chartHoverIndex);

        ctx.beginPath();
        ctx.setLineDash([3, 4]);
        ctx.moveTo(x, top);
        ctx.lineTo(x, height - bottom);
        ctx.strokeStyle = "rgba(235,240,255,0.35)";
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function updateChartHover(event) {
    if (!chartCandles.length) return;

    const canvas = $("price-chart");
    const rect = canvas.getBoundingClientRect();

    const left = 80;
    const right = 16;
    const x = event.clientX - rect.left;

    if (x < left || x > rect.width - right) return;

    const plotWidth = rect.width - left - right;
    const step = plotWidth / chartCandles.length;

    const index = Math.max(
        0,
        Math.min(
            chartCandles.length - 1,
            Math.floor((x - left) / step)
        )
    );

    chartHoverIndex = index;

    const candle = chartCandles[index];
    const tooltip = $("chart-tooltip");

    const date = new Date(candle.timestamp).toLocaleString("ko-KR");

    const rsi = chartIndicators.rsi[index];
    const macd = chartIndicators.macd.macd[index];

    tooltip.textContent = [
        date,
        `O  ${formatPrice(candle.open)}`,
        `H  ${formatPrice(candle.high)}`,
        `L  ${formatPrice(candle.low)}`,
        `C  ${formatPrice(candle.close)}`,
        `VOL  ${candle.volume.toLocaleString("en-US", { maximumFractionDigits: 3 })}`,
        `RSI  ${rsi === null ? "--" : rsi.toFixed(2)}`,
        `MACD  ${macd === null ? "--" : macd.toFixed(3)}`
    ].join("\\n");

    const tooltipWidth = 220;
    const tooltipHeight = 150;

    tooltip.style.left = `${Math.max(
        8,
        Math.min(x + 14, rect.width - tooltipWidth - 8)
    )}px`;

    const pointerY = event.clientY - rect.top;

    tooltip.style.top = `${Math.max(
        8,
        Math.min(pointerY + 12, rect.height - tooltipHeight - 8)
    )}px`;

    tooltip.hidden = false;

    renderTechnicalCharts();
}

function hideChartHover() {
    if (chartHoverIndex === null) return;

    chartHoverIndex = null;
    $("chart-tooltip").hidden = true;

    renderTechnicalCharts();
}

function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number)
        ? `${(number * 100).toFixed(1)}%`
        : "--";
}

function displayBias(value) {
    const bias = String(value || "UNKNOWN").toUpperCase();

    if (bias === "BULLISH") return "LONG";
    if (bias === "BEARISH") return "SHORT";
    if (bias === "NEUTRAL") return "NEUTRAL";

    return bias;
}

function safeText(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function renderReasonList(reasons) {
    if (!Array.isArray(reasons) || reasons.length === 0) {
        return '<li>No reasoning data</li>';
    }

    return reasons.map(reason =>
        `<li>${safeText(reason)}</li>`
    ).join("");
}

async function loadAIAnalysis() {
    const response = await fetch("/api/ai/latest");

    if (!response.ok) {
        throw new Error("AI Analysis API request failed");
    }

    const result = await response.json();

    const noData = $("ai-no-data");
    const content = $("ai-content");
    const status = $("ai-status");

    if (result.status !== "ok" || !result.data) {
        noData.hidden = false;
        content.hidden = true;
        status.textContent = "NO DATA";
        return;
    }

    const data = result.data;

    noData.hidden = true;
    content.hidden = false;
    status.textContent = "ANALYSIS READY";

    const bias = displayBias(data.bias);
    const biasBox = document.querySelector(".ai-bias-box");

    $("ai-bias").textContent = bias;
    $("ai-market").textContent =
        `${data.exchange || "--"} · ${data.symbol || "--"}`;

    biasBox.classList.remove("long", "short", "neutral");

    if (bias === "LONG") biasBox.classList.add("long");
    else if (bias === "SHORT") biasBox.classList.add("short");
    else biasBox.classList.add("neutral");

    $("ai-score").textContent =
        Number.isFinite(Number(data.score))
            ? Number(data.score).toFixed(3)
            : "--";

    $("ai-confidence").textContent =
        formatPercent(data.confidence);

    $("ai-agreement").textContent =
        formatPercent(data.agreement);

    $("ai-regime").textContent =
        safeText(data.regime || "UNKNOWN");

    $("ai-timeframe").textContent =
        safeText(data.timeframe || "--");

    const timestamp = data.generated_at || data.timestamp_ms;

    $("ai-timestamp").textContent = timestamp
        ? (
            data.generated_at
                ? new Date(timestamp).toLocaleString("ko-KR")
                : formatTime(timestamp)
        )
        : "--";

    const reports = Array.isArray(data.reports)
        ? data.reports
        : [];

    $("ai-agents").innerHTML = reports.map(report => {
        const reasons = Array.isArray(report.reasons)
            ? report.reasons
            : [];

        return `
            <article class="ai-agent">
                <div class="ai-agent-header">
                    <strong>${safeText(report.agent_name || "Unknown Agent")}</strong>
                    <span>${safeText(displayBias(report.bias))}</span>
                </div>

                <div class="ai-agent-stats">
                    <span>Score: ${Number.isFinite(Number(report.score)) ? Number(report.score).toFixed(3) : "--"}</span>
                    <span>Confidence: ${formatPercent(report.confidence)}</span>
                </div>

                <ul class="ai-agent-reasons">
                    ${reasons.length
                        ? reasons.map(reason => `<li>${safeText(reason)}</li>`).join("")
                        : "<li>No reasoning data</li>"}
                </ul>
            </article>
        `;
    }).join("");

    $("ai-reasons").innerHTML =
        renderReasonList(data.reasons);
}

async function refreshDashboard() {
    try {
        await Promise.all([
            loadMarket(),
            loadPortfolio(),
            loadHealth(),
            loadAIAnalysis()
        ]);

        $("last-update").textContent =
            new Date().toLocaleTimeString("ko-KR");

    } catch (error) {
        console.error(error);
        $("connection-dot").style.background = "var(--red)";
        $("connection-text").textContent = "Connection Error";
        $("api-status").textContent = "Error";
    }
}

document.querySelectorAll("[data-timeframe]").forEach(button => {
    button.addEventListener("click", () => {
        const timeframe = button.dataset.timeframe;

        if (!TIMEFRAME_LABELS[timeframe]) return;
        if (timeframe === selectedTimeframe) return;

        selectedTimeframe = timeframe;
        chartHoverIndex = null;
        $("chart-tooltip").hidden = true;

        document.querySelectorAll("[data-timeframe]").forEach(item => {
            item.classList.toggle(
                "active",
                item.dataset.timeframe === selectedTimeframe
            );
        });

        loadMarket().catch(error => {
            console.error("Timeframe load failed:", error);
        });
    });
});

document.querySelectorAll("[data-indicator]").forEach(button => {
    button.addEventListener("click", () => {
        const key = button.dataset.indicator;

        if (!(key in indicatorVisibility)) return;

        indicatorVisibility[key] = !indicatorVisibility[key];

        button.classList.toggle("active", indicatorVisibility[key]);

        button.setAttribute(
            "aria-pressed",
            String(indicatorVisibility[key])
        );

        renderTechnicalCharts();
    });
});

$("price-chart").addEventListener("pointermove", updateChartHover);
$("price-chart").addEventListener("pointerleave", hideChartHover);

refreshDashboard();
setInterval(refreshDashboard, 10000);

window.addEventListener("resize", () => {
    loadMarket().catch(console.error);
});

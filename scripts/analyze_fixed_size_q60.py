import csv
import math
from datetime import datetime
from pathlib import Path

INPUT = Path("data/base_vs_1h_macd_q60_trades.csv")

INITIAL_CAPITAL = 10_000.0
FIXED_TRADE_COST = 2_500.0

def number(row, candidates):
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            try:
                result = float(value)
                if math.isfinite(result):
                    return result
            except (TypeError, ValueError):
                pass
    return None


def get_return_pct(row, prefix):
    # 우선 CSV에 기록된 실제 거래 수익률 사용
    value = number(row, [f"{prefix}_return_pct"])
    if value is not None:
        return value

    # 수익률 컬럼이 없는 경우 PnL / 거래 원가로 복원
    pnl = number(row, [f"{prefix}_pnl"])
    quantity = number(row, [f"{prefix}_quantity"])
    entry_price = number(row, [f"{prefix}_entry_price"])
    entry_fee = number(row, [f"{prefix}_entry_fee"])

    if None in (pnl, quantity, entry_price, entry_fee):
        return None

    original_cost = quantity * entry_price + entry_fee
    if original_cost <= 0:
        return None

    return pnl / original_cost * 100


def get_entry_time(row):
    for key in (
        "entry_timestamp_utc",
        "base_entry_timestamp_utc",
        "filtered_entry_timestamp_utc",
    ):
        value = row.get(key)
        if value:
            return value

    return ""


def build_trades(rows, prefix, period):
    trades = []

    for row in rows:
        if row.get("period") != period:
            continue

        return_pct = get_return_pct(row, prefix)
        if return_pct is None:
            continue

        fixed_pnl = FIXED_TRADE_COST * return_pct / 100

        trades.append({
            "timestamp": get_entry_time(row),
            "return_pct": return_pct,
            "fixed_pnl": fixed_pnl,
            "exit_reason": row.get(f"{prefix}_exit_reason", ""),
        })

    # 각 전략은 한 번에 하나의 포지션만 보유하므로
    # 진입 시간 순서가 거래 종료 순서와 동일함.
    trades.sort(key=lambda trade: trade["timestamp"])

    return trades


def calculate_metrics(trades):
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_drawdown = 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0

    for trade in trades:
        pnl = trade["fixed_pnl"]
        equity += pnl

        if pnl > 0:
            gross_profit += pnl
            wins += 1
        elif pnl < 0:
            gross_loss += abs(pnl)

        peak = max(peak, equity)

        if peak > 0:
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

    total_pnl = equity - INITIAL_CAPITAL
    total_return = total_pnl / INITIAL_CAPITAL * 100

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0.0
    )

    win_rate = wins / len(trades) * 100 if trades else 0.0

    atr_stops = sum(
        1 for trade in trades
        if trade["exit_reason"] == "atr_stop"
    )

    return {
        "trades": len(trades),
        "pnl": total_pnl,
        "return": total_return,
        "mdd": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "atr_stops": atr_stops,
        "wins": wins,
    }


def print_metrics(period, name, metrics):
    pf = metrics["profit_factor"]
    pf_text = f"{pf:.3f}" if math.isfinite(pf) else "INF"

    print(
        f"{period:<12} {name:<8}"
        f" Trades={metrics['trades']:>3}"
        f" | PnL={metrics['pnl']:>9.2f}"
        f" | Return={metrics['return']:>7.3f}%"
        f" | Realized MDD={metrics['mdd']:>6.3f}%"
        f" | Win={metrics['win_rate']:>6.2f}%"
        f" | PF={pf_text:>6}"
        f" | ATR Stops={metrics['atr_stops']:>3}"
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"CSV not found: {INPUT}")

    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    print("\n========== FIXED POSITION SIZE EXPERIMENT ==========")
    print(f"Input: {INPUT}")
    print(f"Initial capital: {INITIAL_CAPITAL:.2f} USDT")
    print(f"Fixed trade cost: {FIXED_TRADE_COST:.2f} USDT")
    print("Compounding: DISABLED")
    print("BacktestEngine: NOT MODIFIED")

    required = ["period", "base_pnl", "filtered_pnl"]
    missing = [key for key in required if key not in columns]

    if missing:
        print("\n[ERROR] Required CSV columns are missing:", missing)
        print("Available columns:")
        for column in columns:
            print(" -", column)
        return

    for period in ("TRAIN", "VALIDATION", "FINAL_TEST"):
        base_trades = build_trades(rows, "base", period)
        q60_trades = build_trades(rows, "filtered", period)

        base = calculate_metrics(base_trades)
        q60 = calculate_metrics(q60_trades)

        print(f"\n========== {period} ==========")
        print_metrics(period, "BASE", base)
        print_metrics(period, "Q60", q60)

        print("\n--- Q60 - BASE ---")
        print(f"Trade count: {q60['trades'] - base['trades']:+d}")
        print(f"Fixed-size PnL difference: {q60['pnl'] - base['pnl']:+.2f} USDT")
        print(f"Return difference: {q60['return'] - base['return']:+.3f}%p")
        print(f"Realized MDD difference: {q60['mdd'] - base['mdd']:+.3f}%p")

    print("\n========== EXPERIMENT COMPLETED ==========")
    print("Note: Realized MDD uses closed trades only.")
    print("It is not directly comparable to the engine's mark-to-market MDD.")


if __name__ == "__main__":
    main()

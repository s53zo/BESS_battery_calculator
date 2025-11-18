#!/usr/bin/env python3
"""Simulate filling a BESS from PV exports and selling at spot prices"""
import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from math import ceil

MAX_SELL_PER_INTERVAL_KWH = 4.5  # 18 kW over 15 minutes
DEFAULT_CAPACITY_KWH = 50.0


def read_pv(paths):
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PV data file not found: {path}")
        df = pd.read_csv(
            path,
            usecols=["Časovna značka", "Energija A-"],
            parse_dates=["Časovna značka"],
            dayfirst=False,
        )
        df.rename(columns={"Časovna značka": "timestamp", "Energija A-": "production_kwh"}, inplace=True)
        df["production_kwh"] = pd.to_numeric(df["production_kwh"], errors="coerce").fillna(0.0)
        frames.append(df)
    if not frames:
        raise ValueError("No PV data files were provided.")
    pv = pd.concat(frames, ignore_index=True)
    pv.sort_values("timestamp", inplace=True)
    pv.drop_duplicates("timestamp", inplace=True)
    pv.reset_index(drop=True, inplace=True)
    return pv


def read_price(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price data file not found: {path}")
    df = pd.read_csv(path, skiprows=[0, 2])
    if "Day Ahead Auction (SI)" not in df.columns:
        raise ValueError("Price CSV is missing the 'Day Ahead Auction (SI)' column")
    df = df.dropna(subset=["Day Ahead Auction (SI)", "Date (GMT+1)"])
    df["price_eur_mwh"] = pd.to_numeric(df["Day Ahead Auction (SI)"], errors="coerce")
    df.dropna(subset=["price_eur_mwh"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["Date (GMT+1)"], utc=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Ljubljana")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Ljubljana")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    hourly = df.set_index("timestamp")["price_eur_mwh"].sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    full_index = pd.date_range(start=hourly.index.min(), end=hourly.index.max(), freq="15min")
    price_15min = hourly.reindex(full_index, method="ffill")
    price_15min.name = "price_eur_mwh"
    return price_15min


def simulate_bess(pv_df, price_series, capacity_kwh, max_sell_per_interval_kwh, monthly_windows):
    min_soc = capacity_kwh * 0.10
    max_soc = capacity_kwh
    soc = min_soc
    events = []
    total_revenue = 0.0
    total_sold = 0.0
    lost_energy = 0.0
    monthly_best = {}
    soc_history = []
    pv_df = pv_df.copy()
    pv_df["slot"] = pv_df["timestamp"].dt.hour * 4 + pv_df["timestamp"].dt.minute // 15
    price_df = price_series.rename("price_eur_mwh").to_frame()
    merged = pd.merge(pv_df, price_df, left_on="timestamp", right_index=True, how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps between PV and price records after alignment")
    merged.sort_values("timestamp", inplace=True)
    merged["date"] = merged["timestamp"].dt.date
    grouped = merged.groupby("date", sort=True)
    baseline_revenue = (merged["production_kwh"] * merged["price_eur_mwh"] / 1000.0).sum()

    def record_sale(ts, price, energy_kwh, kind):
        nonlocal total_revenue, total_sold
        revenue = energy_kwh * price / 1000.0
        total_revenue += revenue
        total_sold += energy_kwh
        event = {
            "timestamp": ts,
            "price": price,
            "energy_kwh": energy_kwh,
            "type": kind,
        }
        events.append(event)
        month = ts.month
        best = monthly_best.get(month)
        if best is None or price > best["price"]:
            monthly_best[month] = event
        return event

    def snapshot_state(ts, soc_value, production, price, note=""):
        soc_history.append(
            {
                "timestamp": ts,
                "soc_kwh": soc_value,
                "production_kwh": production,
                "price_eur_mwh": price,
                "note": note,
            }
        )

    for day, day_frame in grouped:
        for _, row in day_frame.iterrows():
            pv_energy = row["production_kwh"]
            note = "idle"
            if pv_energy > 0:
                note = "charging"
                chargeable = max_soc - soc
                charged = min(pv_energy, chargeable)
                soc += charged
                excess = pv_energy - charged
                if excess > 0:
                    sellable = min(excess, max_sell_per_interval_kwh)
                    if sellable > 0:
                        record_sale(row["timestamp"], row["price_eur_mwh"], sellable, "overflow")
                    if excess > max_sell_per_interval_kwh:
                        lost_energy += excess - max_sell_per_interval_kwh
                if soc > max_soc:
                    soc = max_soc
            snapshot_state(row["timestamp"], soc, row["production_kwh"], row["price_eur_mwh"], note)

        discharge_target = max(soc - min_soc, 0.0)
        if discharge_target <= 0:
            continue
        month = day_frame["timestamp"].dt.month.iloc[0]
        window = monthly_windows.get(month)
        if not window:
            continue
        candidates = day_frame[
            (day_frame["slot"] >= window["start_slot"])
            & (day_frame["slot"] <= window["end_slot"])
        ]
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["price_eur_mwh", "timestamp"], ascending=[False, True])
        remaining = discharge_target
        for _, row in candidates.iterrows():
            if remaining <= 0:
                break
            sell_amount = min(max_sell_per_interval_kwh, remaining)
            soc -= sell_amount
            record_sale(row["timestamp"], row["price_eur_mwh"], sell_amount, "discharge")
            remaining -= sell_amount
            snapshot_state(row["timestamp"], soc, row["production_kwh"], row["price_eur_mwh"], "discharge")
        if soc < min_soc:
            soc = min_soc

    return {
        "events": events,
        "monthly_best": monthly_best,
        "total_revenue": total_revenue,
        "total_sold_kwh": total_sold,
        "lost_energy_kwh": lost_energy,
        "soc_history": soc_history,
        "min_soc": min_soc,
        "max_soc": max_soc,
        "baseline_revenue": baseline_revenue,
    }


def format_month(month_number):
    import calendar
    return calendar.month_name[month_number]


def compute_selling_windows(events, lower_quantile=0.05, upper_quantile=0.95):
    df = pd.DataFrame(events)
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    discharge = df[df["type"] == "discharge"].copy()
    if discharge.empty:
        return {}
    discharge["month"] = discharge["timestamp"].dt.month
    discharge["minutes"] = discharge["timestamp"].dt.hour * 60 + discharge["timestamp"].dt.minute
    windows = {}
    for month, group in discharge.groupby("month"):
        lower = group["minutes"].quantile(lower_quantile)
        upper = group["minutes"].quantile(upper_quantile)
        windows[month] = (int(round(lower)), int(round(upper)))
    return windows


def compute_event_windows(events, event_type, lower_quantile=0.05, upper_quantile=0.95):
    df = pd.DataFrame(events)
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    subset = df[df["type"] == event_type].copy()
    if subset.empty:
        return {}
    subset["month"] = subset["timestamp"].dt.month
    subset["minutes"] = subset["timestamp"].dt.hour * 60 + subset["timestamp"].dt.minute
    windows = {}
    for month, group in subset.groupby("month"):
        lower = group["minutes"].quantile(lower_quantile)
        upper = group["minutes"].quantile(upper_quantile)
        windows[month] = (int(round(lower)), int(round(upper)))
    return windows


def slot_to_time(slot):
    hour = slot // 4
    minute = (slot % 4) * 15
    return f"{hour:02d}:{minute:02d}"


def build_monthly_selling_windows(price_series, capacity_kwh, max_sell_per_interval_kwh):
    df = price_series.rename("price_eur_mwh").reset_index()
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["month"] = df["timestamp"].dt.month
    df["slot"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    df = df[df["slot"] < 96]
    min_soc = capacity_kwh * 0.10
    discharge_needed = max(capacity_kwh - min_soc, 0.0)
    intervals_needed = max(1, ceil(discharge_needed / max_sell_per_interval_kwh))
    window_slots = min(intervals_needed, 96)
    windows = {}
    for month, group in df.groupby("month"):
        slot_avg = (
            group.groupby("slot")["price_eur_mwh"]
            .mean()
            .reindex(range(96), fill_value=0.0)
        )
        if window_slots >= 96:
            start = 0
            avg_price = float(slot_avg.mean())
        else:
            best_avg = -float("inf")
            best_start = 0
            for start_slot in range(0, 96 - window_slots + 1):
                avg = float(slot_avg.iloc[start_slot : start_slot + window_slots].mean())
                if avg > best_avg:
                    best_avg = avg
                    best_start = start_slot
            start = best_start
            avg_price = float(
                slot_avg.iloc[start : start + window_slots].mean()
            )
        end = min(start + window_slots - 1, 95)
        windows[month] = {"start_slot": start, "end_slot": end, "avg_price": avg_price}
    return windows


def describe_monthly_windows(windows):
    descriptions = {}
    for month, info in windows.items():
        descriptions[month] = {
            "start": slot_to_time(info["start_slot"]),
            "end": slot_to_time(info["end_slot"]),
            "avg_price": info["avg_price"],
            "slots": info["end_slot"] - info["start_slot"] + 1,
        }
    return descriptions


def average_pv_production(pv_df):
    df = pv_df.copy()
    df["slot"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    slot_avg = df.groupby("slot")["production_kwh"].mean()
    df["production_kwh"] = df["slot"].map(slot_avg)
    df.drop(columns=["slot"], inplace=True)
    return df


def resolve_window_bounds(window):
    if isinstance(window, dict):
        start = window["start_slot"] * 15
        end = window["end_slot"] * 15
        return start, end
    return window


def run_capacity_sweep(
    pv_df,
    price_series,
    min_capacity,
    max_capacity,
    step,
    max_sell_per_interval,
    battery_cost_per_kwh,
):
    if min_capacity <= 0 or max_capacity < min_capacity or step <= 0:
        raise ValueError("Invalid capacity sweep range")
    capacities = []
    c = min_capacity
    while c <= max_capacity + 1e-9:
        capacities.append(round(c, 2))
        c += step
    results = []
    for capacity in capacities:
        windows = build_monthly_selling_windows(price_series, capacity, max_sell_per_interval)
        simulation = simulate_bess(
            pv_df, price_series, capacity, max_sell_per_interval, windows
        )
        revenue = simulation["total_revenue"]
        investment = capacity * battery_cost_per_kwh
        net = revenue - investment
        roi_percent = (net / investment * 100.0) if investment else float("inf")
        payback_years = investment / revenue if revenue > 0 else float("inf")
        results.append(
            {
                "capacity": capacity,
                "revenue": revenue,
                "investment": investment,
                "net": net,
                "roi_percent": roi_percent,
                "payback_years": payback_years,
            }
        )
    return results


def print_sweep_summary(results, objective):
    if not results:
        return
    print(
        f"\nCapacity sweep ({results[0]['capacity']}–{results[-1]['capacity']} kWh) objective={objective}"
    )
    print("cap(kWh)  revenue(€)  investment(€)  net(€)  ROI(%)  payback(years)")
    for row in results:
        print(
            f"{row['capacity']:8.2f}  {row['revenue']:10.2f}  {row['investment']:14.2f}  "
            f"{row['net']:7.2f}  {row['roi_percent']:6.2f}  {row['payback_years']:14.2f}"
        )
    key = "revenue" if objective == "revenue" else "roi_percent"
    best = max(results, key=lambda r: r[key])
    print(
        f"\nBest capacity for {objective}: {best['capacity']} kWh "
        f"(revenue={best['revenue']:.2f} €, ROI={best['roi_percent']:.2f}%, net={best['net']:.2f} €)"
    )


def monthly_window_revenue_from_events(events, windows, event_type):
    if not windows:
        return {}
    df = pd.DataFrame(events)
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
    df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[df["type"] == event_type]
    revenue_info = {}
    for month, window in windows.items():
        if isinstance(window, dict):
            start = window["start_slot"]
            end = window["end_slot"]
        else:
            start, end = window
        month_events = df[df["month"] == month]
        if month_events.empty:
            continue
        window_events = month_events[
            (month_events["minute"] >= start) & (month_events["minute"] <= end)
        ]
        if window_events.empty:
            continue
        daily_energy = (
            window_events.groupby("date")["energy_kwh"]
            .sum()
            .reset_index(name="energy_kwh")
        )
        daily_price = (
            window_events.groupby("date")["price"].mean().reset_index(name="avg_price")
        )
        daily = daily_energy.merge(daily_price, on="date")
        daily["revenue"] = daily["energy_kwh"] * daily["avg_price"] / 1000.0
        revenue_info[month] = {
            "avg_revenue": daily["revenue"].mean(),
            "days": len(daily),
            "avg_energy_kwh": daily["energy_kwh"].mean(),
            "avg_price": daily["avg_price"].mean(),
            "total_revenue": daily["revenue"].sum(),
        }
    return revenue_info


def format_minutes(minutes):
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"


def build_state_df(result, events):
    soc_history = result.get("soc_history", [])
    if not soc_history:
        return pd.DataFrame()
    soc_df = pd.DataFrame(soc_history)
    soc_df["timestamp"] = pd.to_datetime(soc_df["timestamp"])
    min_soc = result.get("min_soc", 0.0)
    max_soc = result.get("max_soc", 0.0)
    if max_soc <= min_soc:
        return pd.DataFrame()

    discharge_events = [
        pd.Timestamp(event["timestamp"]) for event in events if event["type"] == "discharge"
    ]
    last_discharge = {}
    for ts in discharge_events:
        day = ts.date()
        previous = last_discharge.get(day)
        if previous is None or ts > previous:
            last_discharge[day] = ts

    soc_df = soc_df.drop_duplicates(subset="timestamp", keep="last")

    def classify(row):
        eps = 1e-6
        half_soc = max_soc * 0.5
        day = row["timestamp"].date()
        last_ts = last_discharge.get(day)
        if last_ts and row["timestamp"] > last_ts:
            return "Battery empty"
        if row["note"] == "discharge":
            return "Battery emptying for sales"
        if row["note"] == "charging":
            if row["soc_kwh"] >= max_soc - eps:
                return "Battery full"
            if row["soc_kwh"] >= half_soc:
                return "Battery above half capacity"
            return "Battery filling from solar"
        if row["soc_kwh"] <= min_soc + eps:
            return "Battery empty"
        if row["soc_kwh"] >= max_soc - eps:
            return "Battery full"
        if row["soc_kwh"] >= half_soc:
            return "Battery above half capacity"
        return "Battery filling from solar"

    soc_df["state"] = soc_df.apply(classify, axis=1)
    soc_df["date"] = soc_df["timestamp"].dt.date
    return soc_df


def plot_day_detail(day, pv_df, price_series, state_df, min_soc, max_soc, output_dir=Path("bess_charts")):
    day_state = state_df[state_df["date"] == day].sort_values("timestamp")
    if day_state.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    day_start = pd.Timestamp(day)
    day_end = day_start + timedelta(days=1)
    intervals = pd.date_range(day_start, day_end - timedelta(minutes=15), freq="15min")
    detail_df = pd.DataFrame(index=intervals)
    production_series = (
        pv_df.set_index("timestamp")["production_kwh"].reindex(intervals, fill_value=0.0)
    )
    detail_df["production_kwh"] = production_series.fillna(0.0)
    price_day = price_series.reindex(intervals)
    detail_df["price_eur_mwh"] = price_day.ffill().fillna(0.0)

    fig, (ax_state, ax_soc, ax_prod) = plt.subplots(3, 1, sharex=True, figsize=(14, 9))

    state_colors = {
        "Battery empty": "#d62728",
        "Battery filling from solar": "#2ca02c",
        "Battery full": "#9467bd",
        "Battery emptying for sales": "#1f77b4",
        "Battery above half capacity": "#17becf",
    }
    for state, color in state_colors.items():
        subset = day_state[day_state["state"] == state]
        if subset.empty:
            continue
        hours = subset["timestamp"].dt.hour + subset["timestamp"].dt.minute / 60.0
        ax_state.scatter(
            subset["timestamp"],
            hours,
            c=color,
            s=40,
            label=state,
            alpha=0.8,
        )
    ax_state.set_ylim(0, 24)
    ax_state.set_ylabel("Hour of day")
    ax_state.set_title(f"BESS states on {day.isoformat()}")
    ax_state.legend(loc="upper left", fontsize="small")

    ax_soc.plot(day_state["timestamp"], day_state["soc_kwh"], color="#1f77b4", label="SOC")
    ax_soc.axhline(max_soc, color="gray", linestyle="--", label="Capacity")
    ax_soc.axhline(min_soc, color="gray", linestyle=":", label="Min SOC")
    ax_soc.set_ylabel("SOC (kWh)")
    ax_soc.legend(loc="upper left", fontsize="small")

    bar_width = (intervals[1] - intervals[0]).total_seconds() / 86400 * 0.9
    ax_prod.bar(
        detail_df.index,
        detail_df["production_kwh"],
        width=bar_width,
        align="center",
        color="#2ca02c",
        label="PV production",
    )
    ax_prod.set_ylabel("Production (kWh)")
    ax_prod.set_xlabel("Time")
    ax_prod.set_title("Production and price for the selected day")
    ax_price = ax_prod.twinx()
    ax_price.plot(
        detail_df.index,
        detail_df["price_eur_mwh"],
        color="#d62728",
        linewidth=1.5,
        label="Price (EUR/MWh)",
    )
    lines, labels = ax_prod.get_legend_handles_labels()
    lines2, labels2 = ax_price.get_legend_handles_labels()
    ax_prod.legend(lines + lines2, labels + labels2, loc="upper left", fontsize="small")

    fig.tight_layout()
    output_path = output_dir / f"bess_day_{day.isoformat()}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_results(
    result,
    discharge_windows=None,
    overflow_windows=None,
    output_dir=Path("bess_charts"),
    state_df=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = result.get("events", [])
    if not events:
        return
    discharge_windows = discharge_windows or compute_selling_windows(events)
    overflow_windows = overflow_windows or compute_event_windows(events, "overflow")

    events_df = pd.DataFrame(events)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df["date"] = events_df["timestamp"].dt.date

    monthly_best = result.get("monthly_best", {})
    monthly_best_df = (
        pd.DataFrame.from_dict(monthly_best, orient="index")
        .assign(month=lambda df: df.index.astype(int))
        .sort_values("month")
    ) if monthly_best else pd.DataFrame()

    daily_energy = (
        events_df.groupby("date", sort=True)["energy_kwh"]
        .sum()
        .rename("daily_energy")
        .reset_index()
    )

    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("default")

    if not monthly_best_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(
            monthly_best_df["month"],
            monthly_best_df["price"],
            color="steelblue",
            width=0.6,
        )
        ax.set_xticks(monthly_best_df["month"])
        ax.set_xticklabels(
            [format_month(m)[:3] for m in monthly_best_df["month"]],
            rotation=45,
            ha="right",
        )
        ax.set_title("Monthly best-selling price")
        ax.set_xlabel("Month")
        ax.set_ylabel("Price (EUR/MWh)")
        for _, row in monthly_best_df.iterrows():
            ax.annotate(
                f"{row['energy_kwh']:.2f} kWh",
                (row["month"], row["price"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(output_dir / "bess_monthly_best.png", dpi=200)
        plt.close(fig)

    if not daily_energy.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily_energy["date"], daily_energy["daily_energy"], marker="o", ms=3)
        ax.set_title("Daily energy sold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy (kWh)")
        ax.xaxis.set_tick_params(rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "bess_daily_energy.png", dpi=200)
        plt.close(fig)

    window_entries = []
    type_info = [
        ("discharge", discharge_windows, "Discharge", "cornflowerblue"),
        ("overflow", overflow_windows, "Overflow", "goldenrod"),
    ]
    for key, windows, label, _ in type_info:
        for month, window in sorted(windows.items()):
            start, end = resolve_window_bounds(window)
            window_entries.append(
                {"month": month, "start": start, "end": end, "type": key, "label": label}
            )
    if window_entries:
        window_df = pd.DataFrame(window_entries)
        window_df["duration"] = (window_df["end"] - window_df["start"]) / 60.0
        window_df["start_hr"] = window_df["start"] / 60.0
        window_df = window_df.sort_values(["month", "type"])
        y = range(len(window_df))
        type_colors = {t: color for t, _, _, color in type_info}
        fig, ax = plt.subplots(figsize=(10, 4))
        seen_types = set()
        for idx, row in window_df.iterrows():
            color = type_colors[row["type"]]
            label = row["label"] if row["type"] not in seen_types else None
            ax.barh(
                idx,
                row["duration"],
                left=row["start_hr"],
                height=0.6,
                color=color,
                edgecolor="black",
                label=label,
            )
            seen_types.add(row["type"])
        y_labels = [
            f"{format_month(int(row['month']))[:3]} {row['label']}"
            for _, row in window_df.iterrows()
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Hour of day")
        ax.set_title("Typical selling windows per month (5th–95th percentile)")
        ax.set_xlim(0, 24)
        for idx, row in window_df.iterrows():
            ax.text(
                row["start_hr"] + 0.05,
                idx,
                f"{format_minutes(row['start'])}–{format_minutes(row['end'])}",
                va="center",
                fontsize=8,
            )
        handles = [Patch(color=color, label=label) for _, _, label, color in type_info]
        ax.legend(handles=handles, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / "bess_selling_windows.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for kind, subset in events_df.groupby("type"):
        ax.scatter(
            subset["energy_kwh"],
            subset["price"],
            label=kind.capitalize(),
            alpha=0.55,
            s=12,
        )
    ax.set_title("Price versus energy sold")
    ax.set_xlabel("Energy sold per interval (kWh)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bess_price_vs_energy.png", dpi=200)
    plt.close(fig)

    soc_history = result.get("soc_history", [])
    if soc_history:
        soc_df = pd.DataFrame(soc_history)
        soc_df["timestamp"] = pd.to_datetime(soc_df["timestamp"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(soc_df["timestamp"], soc_df["soc_kwh"], color="#1f77b4", linewidth=0.5)
        ax.set_title("Battery state of charge over time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("SOC (kWh)")
        ax.xaxis.set_tick_params(rotation=45)
        if not monthly_best_df.empty:
            for ts in monthly_best_df["timestamp"]:
                ax.axvline(ts, color="orange", alpha=0.3, linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(output_dir / "bess_soc_timeline.png", dpi=200)
        plt.close(fig)

    if state_df is None:
        state_df = build_state_df(result, events)
    if not state_df.empty:
        state_order = [
            "Battery empty",
            "Battery filling from solar",
            "Battery above half capacity",
            "Battery full",
            "Battery emptying for sales",
        ]
        days = state_df["date"].drop_duplicates().sort_values().to_list()
        if days:
            day_index = {day: idx for idx, day in enumerate(days)}
            state_df["day_idx"] = state_df["date"].map(day_index)
            fig, ax = plt.subplots(figsize=(14, 4))
            state_colors = {
                "Battery empty": "#d62728",
                "Battery filling from solar": "#2ca02c",
                "Battery full": "#9467bd",
                "Battery emptying for sales": "#1f77b4",
                "Battery above half capacity": "#17becf",
            }
            for state, color in state_colors.items():
                subset = state_df[state_df["state"] == state]
                if subset.empty:
                    continue
                hours = subset["timestamp"].dt.hour + subset["timestamp"].dt.minute / 60.0
                ax.scatter(
                    subset["day_idx"],
                    hours,
                    c=color,
                    s=3,
                    label=state,
                    alpha=0.6,
                )
            tick_positions = list(range(0, len(days), max(1, len(days) // 12)))
            if tick_positions[-1] != len(days) - 1:
                tick_positions.append(len(days) - 1)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [days[pos].strftime("%Y-%m-%d") for pos in tick_positions],
                rotation=45,
                ha="right",
            )
            ax.set_yticks([0, 6, 12, 18, 24])
            ax.set_ylabel("Hour of day")
            ax.set_title("BESS daily state timeline (per 15-min slot)")
            ax.set_xlabel("Date")
            ax.set_xlim(-0.5, len(days) - 0.5)
            ax.set_ylim(0, 24)
            ax.legend(loc="upper right", fontsize="small")
            fig.tight_layout()
            fig.savefig(output_dir / "bess_state_timeline.png", dpi=200)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BESS calculator using PV exports and day-ahead prices")
    parser.add_argument(
        "--capacity",
        type=float,
        default=DEFAULT_CAPACITY_KWH,
        help="Battery capacity in kWh (default: %(default)s)",
    )
    parser.add_argument(
        "--pv-files",
        nargs="+",
        default=[
            "4-260390-15minMeritve2023-01-01-2023-12-31.csv",
            "4-260390-15minMeritve2024-01-01-2024-12-31.csv",
            "4-260390-15minMeritve2025-01-01-2025-11-17.csv",
        ],
        help="Paths to the PV production CSV files",
    )
    parser.add_argument(
        "--price-file",
        default="energy-charts_Electricity_production_and_spot_prices_in_Slovenia_in_2025.csv",
        help="Path to the day-ahead price CSV",
    )
    parser.add_argument(
        "--max-sell",
        type=float,
        default=MAX_SELL_PER_INTERVAL_KWH,
        help="Maximum energy that can be sold per 15-minute interval in kWh",
    )
    parser.add_argument(
        "--battery-cost-per-kwh",
        type=float,
        default=150.0,
        help="Cost of battery storage per kWh (default: %(default)s EUR/kWh)",
    )
    parser.add_argument(
        "--optimize-capacity",
        action="store_true",
        help="Sweep a range of capacities to find the optimal value for the chosen objective",
    )
    parser.add_argument(
        "--optimize-objective",
        choices=["revenue", "roi"],
        default="revenue",
        help="Objective used during optimization",
    )
    parser.add_argument(
        "--capacity-min",
        type=float,
        default=10.0,
        help="Minimum capacity to test when optimizing (kWh)",
    )
    parser.add_argument(
        "--capacity-max",
        type=float,
        default=100.0,
        help="Maximum capacity to test when optimizing (kWh)",
    )
    parser.add_argument(
        "--capacity-step",
        type=float,
        default=5.0,
        help="Step between tested capacities when optimizing (kWh)",
    )
    parser.add_argument(
        "--one-day",
        metavar="DATE",
        help="Show the per-interval state for a single day (YYYY-MM-DD) and plot the day details",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="Average PV production across all historical data per interval before simulating",
    )
    args = parser.parse_args()

    pv_df = read_pv(args.pv_files)
    if args.average:
        pv_df = average_pv_production(pv_df)
        print("Using averaged PV production per 15-minute interval (all historical data).")
    price_series = read_price(args.price_file)
    monthly_windows = build_monthly_selling_windows(
        price_series, args.capacity, args.max_sell
    )
    result = simulate_bess(
        pv_df, price_series, args.capacity, args.max_sell, monthly_windows
    )
    events = result.get("events", [])
    state_df = build_state_df(result, events)
    window_descriptions = describe_monthly_windows(monthly_windows)

    print(f"Battery capacity: {args.capacity:.1f} kWh")
    print(f"Total sold energy: {result['total_sold_kwh']:.2f} kWh")
    print(f"Total revenue: {result['total_revenue']:.2f} EUR")
    print(f"Total revenue without battery: {result['baseline_revenue']:.2f} EUR")
    print(f"Lost solar energy due to export limits: {result['lost_energy_kwh']:.2f} kWh")
    print("Planned monthly selling window:")
    for month in range(1, 13):
        desc = window_descriptions.get(month)
        if not desc:
            continue
        print(
            f"{format_month(month):>9}: {desc['start']} – {desc['end']} "
            f"({desc['slots']} intervals, avg {desc['avg_price']:.2f} EUR/MWh)"
        )

    selling_windows = compute_selling_windows(events)
    overflow_windows = compute_event_windows(events, "overflow")
    window_revenue_discharge = monthly_window_revenue_from_events(
        result["events"], monthly_windows, "discharge"
    )
    window_revenue_overflow = monthly_window_revenue_from_events(
        result["events"], overflow_windows, "overflow"
    )
    if selling_windows:
        print("Observed discharge windows (5th–95th percentile) per month:")
        for month in sorted(selling_windows):
            start, end = selling_windows[month]
            print(f"  {format_month(month):>9}: {format_minutes(start)} – {format_minutes(end)}")

    chart_dir = Path("bess_charts")
    plot_results(result, monthly_windows, overflow_windows, chart_dir, state_df=state_df)
    print(f"Charts saved to {chart_dir.resolve()}")
    if args.one_day:
        try:
            day = pd.to_datetime(args.one_day).date()
        except (ValueError, TypeError):
            parser.error("--one-day requires a valid date (YYYY-MM-DD)")
        if state_df.empty:
            print(f"No state data available to display for {day.isoformat()}.")
        else:
            day_states = state_df[state_df["date"] == day].sort_values("timestamp")
            if day_states.empty:
                print(f"No intervals recorded for {day.isoformat()}.")
            else:
                print(f"\nDetailed intervals for {day.isoformat()}:")
                if "Battery above half capacity" in day_states["state"].values:
                    print("  Note: 'Battery above half capacity' marks SOC ≥50% but below full.")
                for _, row in day_states.iterrows():
                    time_str = row["timestamp"].strftime("%H:%M")
                    price = row.get("price_eur_mwh")
                    price_str = f"{price:.2f}" if pd.notna(price) else "n/a"
                    print(
                        f"{time_str} | note={row['note']:8} | "
                        f"soc={row['soc_kwh']:5.1f} kWh | state={row['state']} | "
                        f"prod={row['production_kwh']:5.2f} kWh | price={price_str} EUR/MWh"
                    )
                detail_path = plot_day_detail(
                    day,
                    pv_df,
                    price_series,
                    state_df,
                    result["min_soc"],
                    result["max_soc"],
                    chart_dir,
                )
                if detail_path:
                    print(f"Detailed day plot saved to {detail_path.resolve()}")
    if window_revenue_discharge or window_revenue_overflow:
        print("\nPer-day revenue estimates for each selling window type (actual slot prices):")
        if window_revenue_discharge:
            print("  Battery discharge windows:")
            for month in sorted(window_revenue_discharge):
                info = window_revenue_discharge[month]
                print(
                    f"    {format_month(month):>9}: {info['days']} days → "
                    f"{info['avg_energy_kwh']:.2f} kWh/day @ avg {info['avg_price']:.2f} EUR/MWh "
                    f"→ {info['avg_revenue']:.2f} EUR/day ({info['total_revenue']:.2f} EUR/month)"
                )
        if window_revenue_overflow:
            print("  Overflow windows (battery already full):")
            for month in sorted(window_revenue_overflow):
                info = window_revenue_overflow[month]
                print(
                    f"    {format_month(month):>9}: {info['days']} days → "
                    f"{info['avg_energy_kwh']:.2f} kWh/day @ avg {info['avg_price']:.2f} EUR/MWh "
                    f"→ {info['avg_revenue']:.2f} EUR/day ({info['total_revenue']:.2f} EUR/month)"
                )
    if args.optimize_capacity:
        sweep_results = run_capacity_sweep(
            pv_df,
            price_series,
            args.capacity_min,
            args.capacity_max,
            args.capacity_step,
            args.max_sell,
            args.battery_cost_per_kwh,
        )
        print_sweep_summary(sweep_results, args.optimize_objective)
if __name__ == "__main__":
    main()

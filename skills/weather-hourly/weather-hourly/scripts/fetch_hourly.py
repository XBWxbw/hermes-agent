#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather-hourly: 通过 Open-Meteo 获取指定城市指定日期的逐小时天气预报
用法:
  python fetch_hourly.py --city 武汉 --date 2026-05-06
  python fetch_hourly.py --city 北京,深圳 --date tomorrow
  python fetch_hourly.py --lat 30.52 --lon 114.31 --city 武汉 --date today
"""

import argparse
import json
import sys
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("缺少依赖：pip install requests")
    sys.exit(1)

# ── 城市经纬度表 ──────────────────────────────────────────────────────────────
CITIES = {
    "北京": (39.90, 116.40), "上海": (31.23, 121.47),
    "广州": (23.13, 113.26), "深圳": (22.54, 114.06),
    "成都": (30.57, 104.07), "武汉": (30.52, 114.31),
    "南京": (32.06, 118.79), "安陆": (31.26, 113.69),
    "杭州": (30.25, 120.16), "西安": (34.27, 108.95),
    "重庆": (29.56, 106.55), "郑州": (34.75, 113.65),
    "长沙": (28.23, 112.93), "合肥": (31.86, 117.28),
    "苏州": (31.30, 120.62), "天津": (39.13, 117.20),
    "沈阳": (41.79, 123.43), "哈尔滨": (45.75, 126.65),
    "昆明": (25.05, 102.72), "贵阳": (26.58, 106.72),
    "南昌": (28.68, 115.88), "福州": (26.07, 119.30),
    "厦门": (24.48, 118.09), "济南": (36.67, 117.02),
    "青岛": (36.07, 120.38), "大连": (38.91, 121.61),
    "宁波": (29.87, 121.55), "温州": (28.00, 120.67),
    "南宁": (22.82, 108.37), "海口": (20.04, 110.32),
    "三亚": (18.25, 109.51), "乌鲁木齐": (43.82, 87.62),
    "拉萨": (29.65, 91.13),  "银川": (38.47, 106.27),
    "兰州": (36.06, 103.79), "西宁": (36.62, 101.77),
    "呼和浩特": (40.84, 111.75), "太原": (37.87, 112.55),
    "石家庄": (38.04, 114.52), "保定": (38.87, 115.47),
}

# ── WMO 天气代码 ──────────────────────────────────────────────────────────────
WMO_CODES = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "冰粒",
    80: "阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷雨夹冰雹", 99: "强雷雨夹冰雹",
}


def resolve_date(date_str: str) -> str:
    """解析日期字符串，支持 today/tomorrow/YYYY-MM-DD"""
    today = datetime.now().date()
    if date_str.lower() == "today":
        return str(today)
    elif date_str.lower() == "tomorrow":
        return str(today + timedelta(days=1))
    else:
        # 验证格式
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            print(f"错误：日期格式不正确 '{date_str}'，请使用 YYYY-MM-DD、today 或 tomorrow")
            sys.exit(1)


def check_date_range(target_date: str) -> bool:
    """检查目标日期是否在有效预报范围内（今天起16天内）"""
    today = datetime.now().date()
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    delta = (target - today).days
    if delta < 0:
        print(f"⚠ 日期 {target_date} 已过去，Open-Meteo 不提供历史天气（需使用历史API）。")
        return False
    if delta > 15:
        print(f"⚠ 日期 {target_date} 超过16天预报范围（最远可查至 {today + timedelta(days=15)}）。")
        return False
    return True


def fetch_hourly(city_name: str, lat: float, lon: float, target_date: str) -> dict:
    """调用 Open-Meteo API 获取逐小时数据"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "weathercode",
            "precipitation_probability",
            "precipitation",
            "windspeed_10m",
            "relativehumidity_2m",
            "apparent_temperature",
        ],
        "timezone": "Asia/Shanghai",
        "forecast_days": 16,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败 [{city_name}]: {e}")
        return None


def filter_by_date(data: dict, target_date: str) -> list:
    """从完整数据中筛选目标日期的逐小时记录"""
    hourly = data["hourly"]
    result = []
    for i, t in enumerate(hourly["time"]):
        if t.startswith(target_date):
            code = hourly["weathercode"][i]
            result.append({
                "time": t[11:16],  # HH:MM
                "temp": hourly["temperature_2m"][i],
                "feels_like": hourly["apparent_temperature"][i],
                "weather": WMO_CODES.get(code, f"Code:{code}"),
                "rain_prob": hourly["precipitation_probability"][i],
                "precip": hourly["precipitation"][i],
                "wind": hourly["windspeed_10m"][i],
                "humidity": hourly["relativehumidity_2m"][i],
            })
    return result


def format_markdown_table(city_name: str, target_date: str, records: list) -> str:
    """格式化为 Markdown 表格"""
    weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_map[datetime.strptime(target_date, "%Y-%m-%d").weekday()]

    lines = [
        f"## {city_name} · {target_date}（{weekday}）逐小时天气预报",
        "",
        "| 时间  | 天气     | 气温   | 体感温度 | 降雨概率 | 降水量 | 风速      | 湿度 |",
        "|-------|----------|--------|---------|---------|--------|-----------|------|",
    ]
    for r in records:
        lines.append(
            f"| {r['time']} | {r['weather']:<8} | {r['temp']:>4}°C | "
            f"{r['feels_like']:>6}°C | {r['rain_prob']:>5}% | "
            f"{r['precip']:>4}mm | {r['wind']:>6}km/h | {r['humidity']}% |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Open-Meteo 逐小时天气预报")
    parser.add_argument("--city", required=True, help="城市名称（支持逗号分隔多城市）")
    parser.add_argument("--date", required=True, help="日期：YYYY-MM-DD / today / tomorrow")
    parser.add_argument("--lat", type=float, help="自定义纬度（覆盖城市名查表）")
    parser.add_argument("--lon", type=float, help="自定义经度（覆盖城市名查表）")
    parser.add_argument("--output", help="输出文件路径（.md），不指定则打印到控制台")
    args = parser.parse_args()

    target_date = resolve_date(args.date)
    cities = [c.strip() for c in args.city.split(",")]

    all_outputs = [
        f"# 逐小时天气预报",
        f"> 数据来源：Open-Meteo | 查询时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for city_name in cities:
        # 解析经纬度
        if args.lat and args.lon and len(cities) == 1:
            lat, lon = args.lat, args.lon
        elif city_name in CITIES:
            lat, lon = CITIES[city_name]
        else:
            print(f"⚠ 未找到城市 '{city_name}' 的经纬度，请使用 --lat/--lon 参数指定坐标")
            continue

        # 日期范围检查
        if not check_date_range(target_date):
            continue

        print(f"正在获取 {city_name}（{lat},{lon}）{target_date} 的逐小时天气...")
        data = fetch_hourly(city_name, lat, lon, target_date)
        if not data:
            continue

        records = filter_by_date(data, target_date)
        if not records:
            print(f"⚠ {city_name} 在 {target_date} 没有可用数据")
            continue

        table = format_markdown_table(city_name, target_date, records)
        all_outputs.append(table)
        all_outputs.append("")

        # 同时在控制台打印简要版
        print(f"\n{'='*60}")
        print(f"  {city_name} {target_date} 逐小时天气")
        print(f"{'='*60}")
        print(f"{'时间':^6} {'天气':^8} {'气温':^6} {'降雨概率':^8} {'风速':^8} {'湿度':^6}")
        print("-" * 60)
        for r in records:
            print(f"{r['time']:^6} {r['weather']:<8} {r['temp']:>4}°C  {r['rain_prob']:>5}%  {r['wind']:>6}km/h  {r['humidity']}%")

    # 输出文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(all_outputs))
        print(f"\n已保存至：{args.output}")
    else:
        print("\n--- Markdown 输出 ---")
        print("\n".join(all_outputs))


if __name__ == "__main__":
    main()

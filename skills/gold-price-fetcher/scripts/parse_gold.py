#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_gold.py — huangjinjiage.cn 黄金价格数据解析脚本
gold-price-fetcher skill 的数据解析模块

数据来源（双源方案，无需 agent-browser）：
    1. jin.js 实时接口  → 国际/国内贵金属报价（含最高/最低价，实时）
    2. HTML 页面        → 品牌金店报价 + 行情动态（BeautifulSoup 解析）

用法：
    # 直接抓取并解析（默认）
    python parse_gold.py
    python parse_gold.py --format markdown --output result.md
    python parse_gold.py --output result.json

    # 从已有 HTML 文件解析（离线模式）
    python parse_gold.py --input huangjinjiage.html --format markdown

依赖：
    pip install beautifulsoup4
    （requests 为可选加速，未安装时自动用 urllib 替代）

输出：结构化 JSON 或 Markdown 格式数据

数据板块：
    1. 国际黄金价格报价（国际金价、国际银价、国际铂金、国际钯金）
    2. 国内黄金价格报价（国内金价、国内银价）
    3. 十大品牌金店今日金价表
    4. 黄金实时行情动态（最新 3 条）
"""

import re
import json
import argparse
import sys
import time
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# 可选依赖：requests（不安装则用 urllib 替代）
# ─────────────────────────────────────────────

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    import urllib.request
    import urllib.error

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

JINJS_URL = "http://res.huangjinjiage.com.cn/jin.js"
HOME_URL  = "http://www.huangjinjiage.cn"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "http://www.huangjinjiage.cn/",
}

# 内地品牌金店列表
MAINLAND_BRANDS = [
    "老庙黄金", "老凤祥", "周大福", "周生生", "六福珠宝",
    "谢瑞麟", "金至尊", "潮宏基", "菜百首饰", "中国黄金",
    "周六福", "周大生", "恒隆黄金", "豫园股份",
]

# 香港品牌金店列表
HK_BRANDS = [
    "香港周大福", "香港周生生", "香港谢瑞麟", "香港金至尊",
    "香港六福珠宝", "香港老凤祥",
]


# ─────────────────────────────────────────────
# HTTP 工具
# ─────────────────────────────────────────────

def _http_get(url: str, encoding: str = "gbk", timeout: int = 15) -> str:
    """发送 GET 请求，返回解码后的字符串"""
    if _HAS_REQUESTS:
        r = _requests.get(url, headers=_HEADERS, timeout=timeout)
        r.encoding = encoding
        return r.text
    else:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return raw.decode(encoding, errors="replace")


# ─────────────────────────────────────────────
# 数据源 1：jin.js 实时接口
# ─────────────────────────────────────────────

def fetch_jinjs() -> str:
    """获取 jin.js 实时数据（GBK 编码）"""
    return _http_get(JINJS_URL, encoding="gbk")


def _parse_jinjs_var(content: str, var_name: str) -> list[str]:
    """从 jin.js 中提取指定变量的逗号分隔字段"""
    m = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*"([^"]*)"', content)
    return m.group(1).split(",") if m else []


def _jinjs_intl(content: str, var_name: str) -> Optional[dict]:
    """
    解析国际品种字段
    格式：当前价,昨收,开盘,最新价,最高价,最低价,时间,昨收2,前收,?,?,?,日期,品种名,...
    """
    d = _parse_jinjs_var(content, var_name)
    if not d or len(d) < 7:
        return None
    return {
        "最新价": d[3] or d[0],
        "最高价": d[4],
        "最低价": d[5],
        "开盘价": d[2],
        "昨收价": d[7] if len(d) > 7 else d[1],
        "时间":   d[6],
        "日期":   d[12] if len(d) > 12 else "",
    }


def _jinjs_domestic(content: str, var_name: str) -> Optional[dict]:
    """
    解析国内延期品种字段
    格式：当前价,?,开盘,最新价,最高价,最低价,时间,昨收,...,日期,品种名
    """
    return _jinjs_intl(content, var_name)   # 字段位置相同


def _calc_change(latest: str, prev: str) -> tuple[str, str]:
    """根据最新价和昨收价计算涨跌和幅度"""
    try:
        l = float(latest.replace(",", ""))
        p = float(prev.replace(",", ""))
        chg = l - p
        pct = chg / p * 100
        sign = "+" if chg >= 0 else ""
        return f"{sign}{chg:.2f}", f"{sign}{pct:.2f}%"
    except Exception:
        return "—", "—"


def parse_jinjs(content: str) -> tuple[list[dict], list[dict]]:
    """
    解析 jin.js，返回 (国际板块, 国内板块)
    国际板块：国际金价(XAU)、国际银价(XAG)、国际铂金(XPT)、国际钯金(XPD)
    国内板块：黄金延期(AUTD)、白银延期(AGTD)、沪金99(SGE_AU9999)
    """
    intl_map = [
        ("hq_str_hf_XAU",  "国际金价",  "美元/盎司"),
        ("hq_str_hf_XAG",  "国际银价",  "美元/盎司"),
        ("hq_str_hf_XPT",  "国际铂金",  "美元/盎司"),
        ("hq_str_hf_XPD",  "国际钯金",  "美元/盎司"),
    ]
    domestic_map = [
        ("hq_str_gds_AUTD", "黄金延期", "元/克"),
        ("hq_str_gds_AGTD", "白银延期", "元/千克"),
    ]

    # 上金所现货（另外附加）
    sge_map = [
        ("hq_str_SGE_AU9999", "沪金99",   "元/克"),
        ("hq_str_SGE_MAUTD",  "M黄金延期", "元/克"),
    ]

    intl_result = []
    for var, name, unit in intl_map:
        d = _jinjs_intl(content, var)
        if not d:
            continue
        chg, pct = _calc_change(d["最新价"], d["昨收价"])
        intl_result.append({
            "品种": name, "单位": unit,
            "最新价": d["最新价"], "涨跌": chg, "幅度": pct,
            "最高价": d["最高价"], "最低价": d["最低价"],
            "昨收价": d["昨收价"], "时间": d["时间"],
        })

    dom_result = []
    for var, name, unit in domestic_map:
        d = _jinjs_domestic(content, var)
        if not d:
            continue
        chg, pct = _calc_change(d["最新价"], d["昨收价"])
        dom_result.append({
            "品种": name, "单位": unit,
            "最新价": d["最新价"], "涨跌": chg, "幅度": pct,
            "最高价": d["最高价"], "最低价": d["最低价"],
            "昨收价": d["昨收价"], "时间": d["时间"],
        })

    # 上金所现货附加到国内板块
    # 格式：代码,显示名,品种,今开,昨收,最新,最高,最低,...,日期,涨跌幅
    for var, name, unit in sge_map:
        d = _parse_jinjs_var(content, var)
        if not d or len(d) < 8 or d[5] in ("--", ""):
            continue
        latest = d[5]
        prev   = d[4]
        high   = d[6]
        low    = d[7]
        chg, pct = _calc_change(latest, prev)
        # 取 jin.js 自带涨跌幅（更精确）
        if len(d) > 17 and d[17] not in ("NaN%", ""):
            pct = d[17]
        dom_result.append({
            "品种": name, "单位": unit,
            "最新价": latest, "涨跌": chg, "幅度": pct,
            "最高价": high, "最低价": low,
            "昨收价": prev, "时间": d[16][:16] if len(d) > 16 else "",
        })

    return intl_result, dom_result


# ─────────────────────────────────────────────
# 数据源 2：HTML 页面解析
# ─────────────────────────────────────────────

def fetch_html(url: str = HOME_URL) -> str:
    """获取 HTML 页面（GBK 编码）"""
    return _http_get(url, encoding="gbk")


def _bs4_parse_brands(soup) -> dict:
    """用 BeautifulSoup 解析品牌金店价格表
    
    实际页面中品牌名带前缀，如"内地周大福"、"香港周大福"，
    所以直接用 tabtitle class 作为锚点提取所有品牌行，
    而不是用品牌名列表精确匹配。
    """
    mainland = []
    hk = []
    # 跳过非品牌的 tabtitle（价格行情类）
    NON_BRAND_PREFIXES = ("国际", "国内", "投资金条", "黄金回收", "铂金回收", "18K金", "钯金回收")

    def norm(v):
        return "—" if v in ("-", "--", "", "暂无") else v

    for td in soup.find_all("td", class_="tabtitle"):
        raw_name = td.get_text(strip=True)
        # 跳过非品牌行
        if any(raw_name.startswith(p) for p in NON_BRAND_PREFIXES):
            continue

        tr = td.parent
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cells) < 4:
            continue

        gold_p = norm(cells[1]) if len(cells) > 1 else "—"
        plat_p = norm(cells[2]) if len(cells) > 2 else "—"
        bar_p  = norm(cells[3]) if len(cells) > 3 else "—"
        unit   = cells[4] if len(cells) > 4 and cells[4] in ("元/克", "港币/两") else "元/克"

        # 根据前缀判断内地/香港，并去掉"内地"前缀保持品牌名简洁
        is_hk = raw_name.startswith("香港")
        display_name = raw_name.replace("内地", "", 1) if raw_name.startswith("内地") else raw_name

        row = {"品牌": display_name, "黄金价格": gold_p,
               "铂金价格": plat_p, "金条价格": bar_p, "单位": unit}

        if is_hk:
            if not any(r["品牌"] == display_name for r in hk):
                hk.append(row)
        else:
            if not any(r["品牌"] == display_name for r in mainland):
                mainland.append(row)

    def sort_key(r):
        try:
            return float(r.get("黄金价格", "0").replace(",", ""))
        except Exception:
            return 0

    mainland.sort(key=sort_key, reverse=True)
    hk.sort(key=sort_key, reverse=True)
    return {"内地品牌": mainland, "香港品牌": hk}


def _regex_parse_brands(html: str) -> dict:
    """
    正则兜底解析品牌金店（无 bs4 时使用）
    
    实际页面用 class="tabtitle" 标记品牌行，直接以此为锚点提取，
    不依赖品牌名列表匹配（避免"内地周大福"等前缀导致匹配失败）。
    """
    mainland = []
    hk = []
    NON_BRAND_PREFIXES = ("国际", "国内", "投资金条", "黄金回收", "铂金回收", "18K金", "钯金回收")

    # 提取包含 tabtitle 的整行 tr
    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    # 宽松提取 td 内文字（跳过内嵌标签）
    td_text_pattern = re.compile(r'<td[^>]*>\s*(?:<[^>]+>\s*)*([^<\s][^<]*?)(?:\s*<[^>]+>)*\s*</td>', re.DOTALL | re.IGNORECASE)

    def norm(v):
        v = v.strip()
        return "—" if v in ("-", "--", "", "暂无") else v

    for tr_m in tr_pattern.finditer(html):
        tr_html = tr_m.group(1)
        if 'tabtitle' not in tr_html:
            continue

        cells = [m.group(1).strip() for m in td_text_pattern.finditer(tr_html)]
        if len(cells) < 4:
            continue

        raw_name = cells[0]
        # 跳过非品牌行
        if any(raw_name.startswith(p) for p in NON_BRAND_PREFIXES):
            continue

        gold_p = norm(cells[1])
        plat_p = norm(cells[2]) if len(cells) > 2 else "—"
        bar_p  = norm(cells[3]) if len(cells) > 3 else "—"
        unit   = cells[4] if len(cells) > 4 and cells[4] in ("元/克", "港币/两") else "元/克"

        is_hk = raw_name.startswith("香港")
        display_name = raw_name.replace("内地", "", 1) if raw_name.startswith("内地") else raw_name

        row = {
            "品牌": display_name,
            "黄金价格": gold_p,
            "铂金价格": plat_p,
            "金条价格": bar_p,
            "单位": unit,
        }

        if is_hk:
            if not any(r["品牌"] == display_name for r in hk):
                hk.append(row)
        else:
            if not any(r["品牌"] == display_name for r in mainland):
                mainland.append(row)

    def sort_key(r):
        try:
            return float(r.get("黄金价格", "0").replace(",", ""))
        except Exception:
            return 0

    mainland.sort(key=sort_key, reverse=True)
    hk.sort(key=sort_key, reverse=True)
    return {"内地品牌": mainland, "香港品牌": hk}


def _bs4_parse_dynamics(soup, max_items: int = 3) -> list[dict]:
    """用 BeautifulSoup 解析行情动态"""
    results = []
    ts_re = re.compile(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}')

    # 行情动态通常在 class 含 "news"/"dynamic"/"info" 的 div/ul/li 中
    candidates = soup.find_all(
        ["div", "ul", "li", "p"],
        class_=re.compile(r'news|dynamic|info|article|content|list', re.I)
    )
    # 同时搜索含时间戳文字的段落
    for tag in soup.find_all(["li", "p", "div"]):
        text = tag.get_text(strip=True)
        if ts_re.search(text) and len(text) > 30:
            ts_m = ts_re.search(text)
            ts = ts_m.group(0)[:16] if ts_m else "—"
            # 取时间戳之后的内容作为正文
            content = text[ts_m.end():].strip(" :：") if ts_m else text
            if len(content) > 10:
                # 清除内容开头的数字前缀（HTML 中序号/日期残留）
                content = re.sub(r'^\d{1,3}(?=\d{4}年|\d{4}-)', '', content)
                # 去重
                if not any(r["时间"] == ts for r in results):
                    results.append({"时间": ts, "内容": content[:300]})
                    if len(results) >= max_items:
                        break

    return results


def _regex_parse_dynamics(html: str, max_items: int = 3) -> list[dict]:
    """正则兜底解析行情动态"""
    results = []
    ts_re = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})')

    # 去除 HTML 标签
    clean = re.sub(r'<[^>]+>', ' ', html)
    clean = re.sub(r'\s+', ' ', clean)

    matches = list(ts_re.finditer(clean))
    for m in matches[:max_items]:
        ts = m.group(1)[:16]
        # 取时间戳后 400 字符作为内容
        content = clean[m.end():m.end()+400].strip(" :：")
        # 截断到下一个时间戳或句号
        next_ts = ts_re.search(content)
        if next_ts:
            content = content[:next_ts.start()].strip()
        if len(content) > 10:
            results.append({"时间": ts, "内容": content[:300]})

    return results


def parse_html(html: str) -> tuple[dict, list[dict]]:
    """
    解析 HTML，返回 (品牌金店数据, 行情动态列表)
    优先用 BeautifulSoup，否则回退到正则
    """
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        brands   = _bs4_parse_brands(soup)
        dynamics = _bs4_parse_dynamics(soup)
    else:
        brands   = _regex_parse_brands(html)
        dynamics = _regex_parse_dynamics(html)

    return brands, dynamics


# ─────────────────────────────────────────────
# Markdown 输出格式
# ─────────────────────────────────────────────

def to_markdown(data: dict) -> str:
    """将解析结果转换为 Markdown 表格格式"""
    lines = []
    ts = data.get("采集时间", "—")
    lines.append(f"# 黄金价格解析结果 · {ts}\n")

    # 国际黄金价格报价
    lines.append("## 一、国际黄金价格报价\n")
    intl = data.get("国际黄金价格报价", [])
    if intl:
        lines.append("| 品种 | 单位 | 最新价 | 涨跌 | 幅度 | 最高价 | 最低价 | 昨收价 |")
        lines.append("|-----|------|--------|------|------|--------|--------|--------|")
        for r in intl:
            lines.append(
                f"| {r.get('品种','—')} | {r.get('单位','—')} "
                f"| {r.get('最新价','—')} | {r.get('涨跌','—')} | {r.get('幅度','—')} "
                f"| {r.get('最高价','—')} | {r.get('最低价','—')} | {r.get('昨收价','—')} |"
            )
    else:
        lines.append("_未解析到国际黄金数据_")
    lines.append("")

    # 国内黄金价格报价
    lines.append("## 二、国内黄金价格报价\n")
    dom = data.get("国内黄金价格报价", [])
    if dom:
        lines.append("| 品种 | 单位 | 最新价 | 涨跌 | 幅度 | 最高价 | 最低价 | 昨收价 |")
        lines.append("|-----|------|--------|------|------|--------|--------|--------|")
        for r in dom:
            lines.append(
                f"| {r.get('品种','—')} | {r.get('单位','—')} "
                f"| {r.get('最新价','—')} | {r.get('涨跌','—')} | {r.get('幅度','—')} "
                f"| {r.get('最高价','—')} | {r.get('最低价','—')} | {r.get('昨收价','—')} |"
            )
    else:
        lines.append("_未解析到国内黄金数据_")
    lines.append("")

    # 品牌金店
    lines.append("## 三、十大品牌金店今日金价表\n")
    brands = data.get("十大品牌金店今日金价表", {})
    for group_name, unit_hint in [("内地品牌", "元/克"), ("香港品牌", "港币/两")]:
        group = brands.get(group_name, [])
        if group:
            lines.append(f"### {group_name}（{unit_hint}）\n")
            lines.append("| 品牌 | 黄金价格 | 铂金价格 | 金条价格 |")
            lines.append("|-----|--------|--------|--------|")
            for r in group:
                lines.append(
                    f"| {r.get('品牌','—')} | {r.get('黄金价格','—')} "
                    f"| {r.get('铂金价格','—')} | {r.get('金条价格','—')} |"
                )
            lines.append("")
    if not brands.get("内地品牌") and not brands.get("香港品牌"):
        lines.append("_未解析到品牌金店数据_\n")

    # 行情动态
    lines.append("## 四、黄金实时行情动态\n")
    dynamics = data.get("黄金实时行情动态", [])
    if dynamics:
        for item in dynamics:
            lines.append(f"**[{item.get('时间','—')}]** {item.get('内容','')}\n")
    else:
        lines.append("_未解析到行情动态_\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 汇总解析入口
# ─────────────────────────────────────────────

def parse_all(html: str = "", jinjs: str = "") -> dict:
    """
    主解析函数：合并双源数据，返回完整结构化字典

    参数：
        html:  首页 HTML 文本（用于品牌/动态）
        jinjs: jin.js 文本（用于国际/国内实时价格）
    """
    intl_data, dom_data = [], []
    brands_data = {"内地品牌": [], "香港品牌": []}
    dynamics_data = []

    if jinjs:
        intl_data, dom_data = parse_jinjs(jinjs)

    if html:
        brands_data, dynamics_data = parse_html(html)

    return {
        "采集时间":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "国际黄金价格报价":   intl_data,
        "国内黄金价格报价":   dom_data,
        "十大品牌金店今日金价表": brands_data,
        "黄金实时行情动态":   dynamics_data,
    }


# ─────────────────────────────────────────────
# 供外部 import 调用的简化接口
# ─────────────────────────────────────────────

def fetch_and_parse(fmt: str = "dict") -> "dict | str":
    """
    一站式接口：抓取双源数据并返回解析结果

    参数：
        fmt: 返回格式，"dict"（默认）/ "markdown" / "json"

    返回：
        fmt="dict"     → Python dict
        fmt="markdown" → Markdown 字符串
        fmt="json"     → JSON 字符串
    """
    jinjs_text = fetch_jinjs()
    html_text  = fetch_html()
    data = parse_all(html=html_text, jinjs=jinjs_text)
    if fmt == "markdown":
        return to_markdown(data)
    elif fmt == "json":
        return json.dumps(data, ensure_ascii=False, indent=2)
    return data


def parse_from_html(html_text: str, fmt: str = "dict") -> "dict | str":
    """
    离线接口：仅用 HTML 文本解析（无 jin.js，不含实时国际/国内价格）

    参数：
        html_text: 首页 HTML 文本
        fmt: 返回格式，"dict" / "markdown" / "json"
    """
    data = parse_all(html=html_text)
    if fmt == "markdown":
        return to_markdown(data)
    elif fmt == "json":
        return json.dumps(data, ensure_ascii=False, indent=2)
    return data


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="解析 huangjinjiage.cn 黄金价格（HTML + jin.js 双源，无需 agent-browser）"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="离线模式：从已有 HTML 文件解析（不请求网络）"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出文件路径（默认输出到 stdout）"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown"],
        default="json",
        help="输出格式：json（默认）或 markdown"
    )
    args = parser.parse_args()

    if args.input:
        # 离线模式：只解析 HTML
        print(f"[离线模式] 读取文件：{args.input}", file=sys.stderr)
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            html_text = f.read()
        data = parse_all(html=html_text)
    else:
        # 在线模式：双源抓取
        print("[在线模式] 正在抓取 jin.js ...", file=sys.stderr)
        jinjs_text = fetch_jinjs()
        print(f"[jin.js] {len(jinjs_text)} chars", file=sys.stderr)

        print("[在线模式] 正在抓取首页 HTML ...", file=sys.stderr)
        html_text = fetch_html()
        print(f"[HTML] {len(html_text)} chars", file=sys.stderr)

        data = parse_all(html=html_text, jinjs=jinjs_text)

    # 格式化
    if args.format == "markdown":
        output = to_markdown(data)
    else:
        output = json.dumps(data, ensure_ascii=False, indent=2)

    # 写出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[OK] 结果已写入：{args.output}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()

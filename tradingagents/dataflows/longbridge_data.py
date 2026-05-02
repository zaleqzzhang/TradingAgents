"""Longbridge OpenAPI data source for TradingAgents.

Provides stock OHLCV data, technical indicators, and basic fundamentals
via the Longbridge (LongPort) OpenAPI Python SDK.

Required environment variables:
    LONGBRIDGE_APP_KEY
    LONGBRIDGE_APP_SECRET
    LONGBRIDGE_ACCESS_TOKEN

Install: pip install longbridge
"""

import os
from datetime import datetime, timedelta
from typing import Optional

_ctx_cache: dict = {}


class LongbridgeRateLimitError(Exception):
    """Raised when Longbridge API rate limit is hit."""
    pass


def _get_quote_context():
    """Get or create a cached QuoteContext."""
    if "quote" not in _ctx_cache:
        from longbridge.openapi import Config, QuoteContext
        config = Config.from_env()
        _ctx_cache["quote"] = QuoteContext(config)
    return _ctx_cache["quote"]


def _format_symbol(symbol: str) -> str:
    """Convert symbol to Longbridge format.

    Longbridge uses: 700.HK, AAPL.US, 600519.SH, 000001.SZ
    If no suffix is provided, default to .US for US stocks.
    """
    if "." in symbol:
        return symbol.upper()
    # Default to US market
    return f"{symbol.upper()}.US"


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Get OHLCV stock data from Longbridge.

    Args:
        symbol: Ticker symbol (e.g., NVDA, 700.HK, AAPL.US)
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing daily OHLCV data.
    """
    from longbridge.openapi import Period, AdjustType

    ctx = _get_quote_context()
    lb_symbol = _format_symbol(symbol)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate number of trading days needed (approximate)
    days_diff = (end_dt - start_dt).days
    count = min(max(days_diff * 2, 100), 1000)  # Request enough data

    try:
        candlesticks = ctx.candlesticks(lb_symbol, Period.Day, count, AdjustType.ForwardAdj)
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "limit" in error_msg.lower():
            raise LongbridgeRateLimitError(f"Longbridge rate limited: {error_msg}")
        raise

    if not candlesticks:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Filter by date range and format as CSV
    lines = ["timestamp,open,high,low,close,volume,turnover"]
    for candle in candlesticks:
        candle_date = candle.timestamp.date() if hasattr(candle.timestamp, 'date') else candle.timestamp
        if isinstance(candle_date, str):
            candle_date = datetime.strptime(candle_date[:10], "%Y-%m-%d").date()
        elif hasattr(candle_date, 'date'):
            candle_date = candle_date.date()

        if start_dt.date() <= candle_date <= end_dt.date():
            lines.append(
                f"{candle_date},{candle.open},{candle.high},"
                f"{candle.low},{candle.close},{candle.volume},{candle.turnover}"
            )

    if len(lines) == 1:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    header = f"# Stock data for {lb_symbol} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(lines) - 1}\n"
    header += f"# Data source: Longbridge OpenAPI\n\n"

    return header + "\n".join(lines)


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
) -> str:
    """Calculate technical indicators from Longbridge candlestick data.

    Uses stockstats library to compute indicators from raw OHLCV data.

    Args:
        symbol: Ticker symbol
        indicator: Technical indicator name
        curr_date: Current trading date (YYYY-mm-dd)
        look_back_days: How many days to look back
        interval: Time interval (daily)
        time_period: Number of data points for calculation
        series_type: Price type (close, open, high, low)

    Returns:
        String containing indicator values and description.
    """
    from longbridge.openapi import Period, AdjustType
    import pandas as pd

    indicator_descriptions = {
        "close_50_sma": "50 SMA: A medium-term trend indicator.",
        "close_200_sma": "200 SMA: A long-term trend benchmark.",
        "close_10_ema": "10 EMA: A responsive short-term average.",
        "macd": "MACD: Computes momentum via differences of EMAs.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line.",
        "macdh": "MACD Histogram: Shows the gap between MACD and signal.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold conditions.",
        "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands.",
        "boll_ub": "Bollinger Upper Band: 2 std deviations above the middle.",
        "boll_lb": "Bollinger Lower Band: 2 std deviations below the middle.",
        "atr": "ATR: Averages true range to measure volatility.",
        "vwma": "VWMA: A moving average weighted by volume.",
    }

    ctx = _get_quote_context()
    lb_symbol = _format_symbol(symbol)

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    # Fetch extra data for indicator warmup
    extra_days = max(look_back_days + 250, 400)
    count = min(extra_days, 1000)

    try:
        candlesticks = ctx.candlesticks(lb_symbol, Period.Day, count, AdjustType.ForwardAdj)
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "limit" in error_msg.lower():
            raise LongbridgeRateLimitError(f"Longbridge rate limited: {error_msg}")
        raise

    if not candlesticks:
        return f"No data available for {symbol} to compute {indicator}"

    # Build DataFrame
    records = []
    for candle in candlesticks:
        candle_date = candle.timestamp
        if hasattr(candle_date, 'date'):
            candle_date = candle_date.date()
        elif isinstance(candle_date, str):
            candle_date = datetime.strptime(candle_date[:10], "%Y-%m-%d").date()
        records.append({
            "date": candle_date,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": int(candle.volume),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)

    # Use stockstats to compute indicators
    from stockstats import StockDataFrame
    stock_df = StockDataFrame.retype(df.copy())

    try:
        indicator_values = stock_df[indicator]
    except Exception as e:
        return f"Error computing {indicator}: {str(e)}"

    # Filter to the requested date range
    before = curr_date_dt - timedelta(days=look_back_days)
    result_lines = []
    for idx, row in df.iterrows():
        row_date = row["date"]
        if isinstance(row_date, str):
            row_date = datetime.strptime(row_date, "%Y-%m-%d").date()
        if before.date() <= row_date <= curr_date_dt.date():
            val = indicator_values.iloc[idx] if idx < len(indicator_values) else None
            if val is not None and pd.notna(val):
                result_lines.append(f"{row_date}: {val:.4f}")

    if not result_lines:
        ind_string = "No data available for the specified date range.\n"
    else:
        ind_string = "\n".join(result_lines)

    desc = indicator_descriptions.get(indicator, "No description available.")
    return (
        f"## {indicator.upper()} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        f"{ind_string}\n\n{desc}"
    )


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Get basic stock information from Longbridge.

    Args:
        ticker: Ticker symbol
        curr_date: Current date (not used)

    Returns:
        String with basic stock info.
    """
    ctx = _get_quote_context()
    lb_symbol = _format_symbol(ticker)

    try:
        info_list = ctx.static_info([lb_symbol])
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "limit" in error_msg.lower():
            raise LongbridgeRateLimitError(f"Longbridge rate limited: {error_msg}")
        return f"Error retrieving fundamentals for {ticker}: {error_msg}"

    if not info_list:
        return f"No fundamentals data found for symbol '{ticker}'"

    info = info_list[0]

    # Also get real-time quote for additional data
    try:
        quotes = ctx.quote([lb_symbol])
        quote = quotes[0] if quotes else None
    except Exception:
        quote = None

    lines = [f"# Company Info for {lb_symbol}"]
    lines.append(f"# Data source: Longbridge OpenAPI\n")

    # Static info fields
    for attr in ["name_cn", "name_en", "name_hk", "exchange", "currency",
                 "lot_size", "total_shares", "circulating_shares",
                 "board", "security_type"]:
        val = getattr(info, attr, None)
        if val is not None:
            lines.append(f"{attr}: {val}")

    # Quote data
    if quote:
        for attr in ["last_done", "prev_close", "open", "high", "low",
                     "volume", "turnover", "timestamp"]:
            val = getattr(quote, attr, None)
            if val is not None:
                lines.append(f"{attr}: {val}")

        # Calculate change
        if quote.last_done and quote.prev_close and float(quote.prev_close) > 0:
            change_pct = (float(quote.last_done) - float(quote.prev_close)) / float(quote.prev_close) * 100
            lines.append(f"change_pct: {change_pct:.2f}%")

    return "\n".join(lines)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Longbridge does not provide balance sheet data directly.
    Falls back to basic quote info with a note.
    """
    return (
        f"Balance sheet data is not available via Longbridge OpenAPI for {ticker}.\n"
        f"Longbridge focuses on real-time market data and trading.\n"
        f"Consider using Alpha Vantage or yfinance for fundamental financial statements."
    )


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Longbridge does not provide cash flow data directly."""
    return (
        f"Cash flow data is not available via Longbridge OpenAPI for {ticker}.\n"
        f"Consider using Alpha Vantage or yfinance for financial statements."
    )


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Longbridge does not provide income statement data directly."""
    return (
        f"Income statement data is not available via Longbridge OpenAPI for {ticker}.\n"
        f"Consider using Alpha Vantage or yfinance for financial statements."
    )


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Longbridge does not provide a news API endpoint.
    Returns a note suggesting alternative sources.
    """
    return (
        f"News data is not directly available via Longbridge OpenAPI for {ticker}.\n"
        f"Longbridge focuses on real-time market data and trading.\n"
        f"Consider using Alpha Vantage or yfinance for news sentiment data."
    )


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 50) -> str:
    """Longbridge does not provide a global news API."""
    return (
        f"Global news data is not available via Longbridge OpenAPI.\n"
        f"Consider using Alpha Vantage or yfinance for global news."
    )


def get_insider_transactions(symbol: str) -> str:
    """Longbridge does not provide insider transaction data."""
    return (
        f"Insider transaction data is not available via Longbridge OpenAPI for {symbol}.\n"
        f"Consider using Alpha Vantage or yfinance for insider data."
    )

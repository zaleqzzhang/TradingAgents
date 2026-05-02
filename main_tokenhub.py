"""TradingAgents with Tencent Cloud TokenHub (OpenAI-compatible).

Usage:
  1. Fill in your .env file:
       TOKENHUB_API_KEY=your-api-key
       TOKENHUB_BASE_URL=https://your-endpoint/v1
       TOKENHUB_MODEL=your-model-id

  2. Run:
       python main_tokenhub.py
"""

import os

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()

# Read TokenHub config from environment
tokenhub_api_key = os.environ.get("TOKENHUB_API_KEY", "")
tokenhub_base_url = os.environ.get("TOKENHUB_BASE_URL", "")
tokenhub_model = os.environ.get("TOKENHUB_MODEL", "")

if not tokenhub_api_key:
    raise ValueError("TOKENHUB_API_KEY is not set in .env")
if not tokenhub_base_url:
    raise ValueError("TOKENHUB_BASE_URL is not set in .env")
if not tokenhub_model:
    raise ValueError("TOKENHUB_MODEL is not set in .env")

# Configure TradingAgents to use TokenHub via OpenAI-compatible mode
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["backend_url"] = tokenhub_base_url
config["deep_think_llm"] = tokenhub_model
config["quick_think_llm"] = tokenhub_model
config["max_debate_rounds"] = 1
config["output_language"] = "Chinese"

# Use Longbridge for market data, Alpha Vantage as fallback for fundamentals/news
config["data_vendors"] = {
    "core_stock_apis": "longbridge",          # Longbridge for OHLCV (HK/US/A-shares)
    "technical_indicators": "longbridge",     # Longbridge for indicators
    "fundamental_data": "longbridge",         # Basic info from Longbridge, fallback to others
    "news_data": "alpha_vantage",             # Longbridge has no news API, use AV
}

# Override OPENAI_API_KEY so the OpenAI client picks it up
os.environ["OPENAI_API_KEY"] = tokenhub_api_key

# Initialize and run
ta = TradingAgentsGraph(debug=True, config=config)

# Example: analyze NVDA on a specific date
_, decision = ta.propagate("0700.HK", "2026-05-02")
print(decision)

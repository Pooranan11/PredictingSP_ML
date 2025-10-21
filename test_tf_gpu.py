import finnhub
import os
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=api_key)

now = int(time.time())
past = now - 30 * 24 * 60 * 60  # 30 jours

res = finnhub_client.stock_candles("NVDA", "D", past, now)
print(res)

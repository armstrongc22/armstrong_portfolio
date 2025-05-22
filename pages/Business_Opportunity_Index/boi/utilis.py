import requests, requests_cache, time
from tenacity import retry, wait_exponential, stop_after_attempt

# 7-day cache → slashes API quota
requests_cache.install_cache("http_cache", expire_after=7*24*3600)

@retry(wait=wait_exponential(multiplier=1, min=2, max=60),
       stop=stop_after_attempt(6))
def safe_get(url, **kw):
    r = requests.get(url, **kw)
    if r.status_code == 429:
        raise Exception("Rate-limited")
    r.raise_for_status()
    return r
"""
Probe current Groq quota state for all three models by sending a 
minimal 1-token completion and reading response headers.
Reports: x-ratelimit-remaining-tokens, x-ratelimit-reset-tokens,
         x-ratelimit-remaining-requests, x-ratelimit-reset-requests
         x-ratelimit-limit-tokens-per-day (if present)
"""
import os, json, time, requests
from datetime import datetime, timezone

API_KEY = os.environ.get("GROQ_API_KEY", "")
if not API_KEY:
    # Try to load from .env or harness config
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get("GROQ_API_KEY", "")
    except ImportError:
        pass

if not API_KEY:
    # Try reading from harness config file
    try:
        import sys
        sys.path.insert(0, "harness")
        from groq_client import GroqClient
        c = GroqClient.__new__(GroqClient)
        # If key is embedded
    except Exception:
        pass

MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

RATE_HEADER_KEYS = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-limit-tokens-per-day",
    "x-ratelimit-remaining-tokens-per-day",
    "x-ratelimit-reset-tokens-per-day",
    "retry-after",
    "x-ratelimit-limit-requests-per-day",
    "x-ratelimit-remaining-requests-per-day",
]

now = datetime.now(timezone.utc).isoformat()
print(f"Quota probe at {now}\n{'='*60}")

results = {}
for model in MODELS:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
        headers = dict(resp.headers)
        body = resp.json()
        
        print(f"\n--- {model} ---")
        print(f"  HTTP status: {resp.status_code}")
        
        quota_info = {}
        for k in RATE_HEADER_KEYS:
            v = headers.get(k) or headers.get(k.lower())
            if v:
                quota_info[k] = v
                print(f"  {k}: {v}")
        
        if resp.status_code == 429:
            print(f"  RATE LIMITED — body: {json.dumps(body)[:400]}")
        elif resp.status_code == 200:
            usage = body.get("usage", {})
            print(f"  usage: {usage}")
        else:
            print(f"  body: {json.dumps(body)[:400]}")
        
        results[model] = {
            "status": resp.status_code,
            "quota_headers": quota_info,
            "usage": body.get("usage") if resp.status_code == 200 else None,
            "error": body.get("error") if resp.status_code != 200 else None,
        }
        
    except Exception as e:
        print(f"\n--- {model} ---")
        print(f"  ERROR: {e}")
        results[model] = {"error": str(e)}
    
    time.sleep(1)  # don't hammer in quick succession

print(f"\n{'='*60}")
print("Summary JSON:")
print(json.dumps(results, indent=2))

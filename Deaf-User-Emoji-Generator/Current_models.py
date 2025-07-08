import os
import requests
import json

# ✅ Replace this with your actual API key
api_key = "gsk_your_actual_key_here"

url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

try:
    data = response.json()

    # DEBUG print to see structure
    print("📦 Full JSON Response:\n", json.dumps(data, indent=2))

    # Check if "data" key exists and is a list
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        print("\n✅ Supported Groq Models:")
        for model in data["data"]:
            if isinstance(model, dict) and "id" in model:
                print("–", model["id"])
    else:
        print("❌ 'data' key missing or not a list in Groq API response.")

except Exception as e:
    print("❌ Error while parsing JSON:", str(e))

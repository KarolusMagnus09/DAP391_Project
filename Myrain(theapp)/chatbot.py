import requests
import re

# API KEYS
MISTRAL_API_KEY = "DKOA5AwEgZEc6FP8FBEyYSQyYFj3j9bz"
OPENWEATHER_API_KEY = "ed871980da04670196858eeec8601527"

# ENDPOINTS
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

MODEL = "mistral-small"

conversation = []

def get_weather(city_name):
    params = {
        "q": city_name,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        res = requests.get(OPENWEATHER_URL, params=params)
        data = res.json()

        if res.status_code != 200:
            return f" Could not find weather for '{city_name}'."

        temp = data['main']['temp']
        desc = data['weather'][0]['description'].capitalize()
        humid = data['main']['humidity']
        wind = data['wind']['speed']

        return (f" Weather in {city_name.title()}:\n"
                f"- Temperature: {temp}°C\n"
                f"- Condition: {desc}\n"
                f"- Humidity: {humid}%\n"
                f"- Wind Speed: {wind} m/s")
    except Exception as e:
        return f"⚠️ Failed to get weather: {e}"

def detect_city_query(text):
    match = re.search(r"(?:weather|temperature).*in ([A-Za-z\s]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

print("Myrain: Hi! Ask me anything — or type something like 'weather in Hanoi'. Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Myrain: Bye! Stay safe")
        break

    city = detect_city_query(user_input)
    if city:
        print("Myrain: Fetching weather for", city, "...\n")
        weather_report = get_weather(city)
        print("Myrain:", weather_report, "\n")
        continue

    conversation.append({"role": "user", "content": user_input})
    payload = {
        "model": MODEL,
        "messages": conversation,
        "temperature": 0.7
    }

    try:
        res = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
        print("Myrain:", reply, "\n")
        conversation.append({"role": "assistant", "content": reply})
    except Exception as e:
        print(" Error:", e)

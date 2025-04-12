from model import ChatbotNN
import json, random, torch, numpy as np, nltk, re, requests, pickle
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import os

nltk.download("punkt")
nltk.download("wordnet")

load_dotenv()
weather_api = os.getenv('weather_api')

with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
all_words = data["all_words"]
tags = data["tags"]
intents_data = data["intents_data"]

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(sentence):
    words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in words]

def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

def get_lat_lon(city):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={weather_api}"
    response = requests.get(geo_url).json()
    if response:
        return response[0]['lat'], response[0]['lon']
    return None, None

def get_weather(city):
    lat, lon = get_lat_lon(city)
    if lat is None or lon is None:
        return "City not found!"
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        weather = data["weather"][0]["description"]
        return f"The weather in {city.capitalize()} is {weather} with a temperature of {temp - 273.15:.2f}°C."
    return "Couldn't fetch weather data."

def get_time(city):
    url = "http://worldtimeapi.org/api/timezone"
    response = requests.get(url).json()
    for tz in response:
        if city.lower() in tz.lower():
            time_response = requests.get(f"http://worldtimeapi.org/api/timezone/{tz}").json()
            if "datetime" in time_response:
                return f"The current time in {city.capitalize()} is {time_response['datetime'][:19].replace('T', ' ')}."
    return "Couldn't fetch time data."

def process_message(input_message):
    tokenized_sentence = tokenize_and_lemmatize(input_message)
    bag = bag_of_words(tokenized_sentence, all_words)
    bag_tensor = torch.tensor([bag], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(bag_tensor)

    predicted_class_index = torch.argmax(predictions, dim=1).item()
    predicted_intent = tags[predicted_class_index]

    if predicted_intent == "weather":
        city_match = re.search(r'in (\w+)', input_message.lower())
        if city_match:
            city = city_match.group(1)
            return get_weather(city)
        return "Please specify a city, e.g., 'What's the weather in Delhi?'"

    if predicted_intent == "time":
        city_match = re.search(r'in (\w+)', input_message.lower())
        if city_match:
            city = city_match.group(1)
            return get_time(city)
        return "Please specify a city, e.g., 'What time is it in New York?'"

    for intent in intents_data["intents"]:
        if intent["tag"] == predicted_intent:
            return random.choice(intent["responses"])

    return "I'm not sure how to respond to that."

# Install required libraries silently
!pip install -q transformers faiss-cpu sentence-transformers requests gradio google-genai

# Imports
import requests
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr
from datetime import datetime, timedelta
import re
from google import genai

"""## API Setup and configuration"""

WEATHER_API_KEY = "c29a1c1db2f59aee082e6dab1c16debc"
GEMINI_API_KEY = "AIzaSyCmtFUG1DN4HVQm1asEMmOGoN3UIB8YopU"

# OpenWeatherMap 5-day / 3-hour forecast endpoint
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_CONTEXT = 3  # Number of relevant documents to retrieve in semantic search
FORECAST_DAYS = 5  # Display forecast days

"""## Creating the class for predicting the weather using RAG"""

class WeatherRAGAssistant:

    ## Initializing the Class by a constructor ##
    def __init__(self, weather_key: str, gemini_key: str):
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.weather_key = weather_key
        self.index = None
        self.texts = []  # Stores all formatted weather chunks
        self.api_response = None # Store the API response for timezone handling

    ## Fetching and Indexing Weather ##
    def fetch_and_index_weather(self, location: str, days: int = FORECAST_DAYS):
        params = {"q": location, "appid": self.weather_key, "units": "metric"}  # Celsius
        response = requests.get(WEATHER_BASE_URL, params=params, timeout=10)
        response.raise_for_status()  # Raises exception if response has error (e.g., 404 city not found)
        api_response = response.json()  # Converts API response (JSON) into a Python dictionary
        self.api_response = api_response # Store the API response for timezone alignment

        # _format_weather_data() → processes raw API data → clean daily summaries.
        self.texts = self._format_weather_data(api_response)
        # _build_faiss_index() → turns summaries into embeddings + stores in FAISS.
        self._build_faiss_index(self.texts)

        return api_response, self.texts

    ## Formatting Weather Data ##
    def _format_weather_data(self, api_response: dict) -> list[str]:
        daily_data = {}
        timezone_offset = api_response['city'].get('timezone', 0) # Get timezone offset in seconds

        for three_hour_forecast in api_response.get('list', []):  # Iterate over the 3-hour forecasts to aggregate daily values
            timestamp = three_hour_forecast['dt']
            utc_datetime = datetime.utcfromtimestamp(timestamp)
            local_datetime = utc_datetime + timedelta(seconds=timezone_offset)  # Adjust with city timezone
            date_str = local_datetime.strftime('%Y-%m-%d')

            # Extract main metrics
            temp_max = three_hour_forecast['main']['temp_max']
            temp_min = three_hour_forecast['main']['temp_min']
            description = three_hour_forecast['weather'][0]['description']
            rain = three_hour_forecast.get('rain', {}).get('3h', 0)

            if date_str not in daily_data:   # If the date is seen first time then initialize dictionary for that day.
                daily_data[date_str] = {
                    'temp_max': temp_max, 'temp_min': temp_min,
                    'description_list': set(), 'rain_total': 0.0
                }

            # Update min/max average for each day
            daily_data[date_str]['temp_max'] = max(daily_data[date_str]['temp_max'], temp_max)
            daily_data[date_str]['temp_min'] = min(daily_data[date_str]['temp_min'], temp_min)
            daily_data[date_str]['description_list'].add(description)
            daily_data[date_str]['rain_total'] += rain

        texts = []          # Create a neat and clean summary string for the day
        for date, data in daily_data.items():
            descriptions = ", ".join(sorted(list(data['description_list'])))
            rain_total_rounded = round(data['rain_total'], 2)

            texts.append(
                f"Date: {date}, Max Temp: {round(data['temp_max'], 1)}°C, Min Temp: {round(data['temp_min'], 1)}°C, "
                f"Total Rain: {rain_total_rounded}mm, Conditions: {descriptions}"
            )

        texts.sort()  # sorting by date
        return texts

    ## Build FAISS index ##
    def _build_faiss_index(self, texts: list[str]):  # Turns summaries into numerical vectors (embeddings).
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)  # converting from text to embeddings
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))  # Ensures data format is FAISS compatible

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)  # L2 (Euclidean) distance index
        self.index.add(embeddings)

    ## Search Index ##
    def search_index(self, query: str) -> list[str]:
        if self.index is None or not self.texts:
            return []

        try:
            query_vec = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')   # Converts query into embedding
            D, I = self.index.search(query_vec, TOP_K_CONTEXT) # Searches FAISS index where D = distances and I = Indices

            context = [self.texts[i] for i in I[0] if 0 <= i < len(self.texts)]  # Picks matching summary
            return context   # returns list of summary items

        except Exception as e:
            return []

    ## Generate Response with Gemini ##
    def generate_response(self, user_query: str, context: list[str]) -> str:   # Combines user’s question + weather context → asks Gemini for final answer.
        if not context:
            return "I could not find relevant weather data."

        prompt = f"""
        You are a helpful weather assistant.
        Context:
        ---
        {'\n'.join(context)}
        ---
        Question: {user_query}
        Answer:
        """

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Gemini API Error: Could not generate response. Details: {e}"

"""## RAG Pipeline"""

assistant = WeatherRAGAssistant(WEATHER_API_KEY, GEMINI_API_KEY)

## Gradio will start here as it is UI interface for the user ##
initial_location_texts = []  # it will store the every day weather summaries
INITIAL_PROMPT = f"Enter location or use 📍 Use My Location to load the {FORECAST_DAYS}-day forecast." # display

## how to give the location to the API ##
def reverse_geocode(lat: float, lon: float) -> str:
    try:
        url = "http://api.openweathermap.org/geo/1.0/reverse"
        params = {"lat": lat, "lon": lon, "limit": 1, "appid": WEATHER_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            city = data[0].get("name", "")
            country = data[0].get("country", "")
            return f"{city}, {country}"
        return "Unknown Location"
    except Exception as e:
        return f"Error: {e}"

def use_my_location(lat, lon):
    global initial_location_texts
    location_name = reverse_geocode(lat, lon)
    _, initial_location_texts = assistant.fetch_and_index_weather(location=location_name)
    if initial_location_texts:
        return f"📍 Location detected: {location_name}. Weather data loaded!"
    else:
        initial_location_texts = []
        return f"❌ Failed to load weather data for: {location_name}"

def update_location_and_index(new_location: str):
    global initial_location_texts
    clean_location = new_location.strip()

    # _. used to avoid the raw data from API
    _, initial_location_texts = assistant.fetch_and_index_weather(location=clean_location) # returns 2 things, API_response and texts

    if initial_location_texts:
        return f"Weather data for {clean_location} loaded!"
    else:
        initial_location_texts = []
        return f"❌ Failed to load weather data for {clean_location}."

## Understanding the user's string data ##
def get_target_date_string(user_query: str) -> str | None:
    """Helper function to convert relative time (Today, Tomorrow, Friday) into YYYY-MM-DD format using city timezone."""
    query_lower = user_query.lower() # converting into lower case

    # Align with city timezone
    tz_offset = assistant.api_response['city'].get('timezone', 0) if assistant.api_response else 0
    current_time = datetime.utcnow() + timedelta(seconds=tz_offset)
    target_date = None  # Recognizes the date

    day_mapping = {  # A small lookup table saying how many days from now those relative words mean
        'today': 0,
        'tomorrow': 1,
        'day after tomorrow': 2,
    }

    current_weekday = current_time.weekday()   # Check Day Names (e.g., 'friday')
    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    for i in range(7):  # will check and count what the user mentions as a day
        target_day_index = (current_weekday + i) % 7
        if day_names[target_day_index] in query_lower:
            target_date = current_time + timedelta(days=i)
            break

    for term, days_delta in day_mapping.items():   # Check Relative Terms like today, tomorrow, day after tomorrow
        if term in query_lower:
            target_date = current_time + timedelta(days=days_delta)
            break

    if target_date:
        return target_date.strftime('%Y-%m-%d')   # If a date was found, convert it to the YYYY-MM-DD string format and return it.

    return None

## User to UI and UI to model ##
def answer_weather_query(user_query: str):
    if not initial_location_texts:
        return "Please load the weather data first."  # load the location first

    target_date_str = get_target_date_string(user_query)  # parse the query data to gather the relative terms

    if target_date_str:     # find the summary that contains the date string in the index
        target_data = None
        for text in initial_location_texts:
            if target_date_str in text:
                target_data = text
                break

        if target_data is None:
            return f"The forecast data for {target_date_str} is not available in the loaded {len(initial_location_texts)}-day forecast range."

        final_context = f"The verified weather data for {target_date_str} is: {target_data}"   # Builds a short single-line context string that clearly states which date and which data is being used. This will be fed to the AI.
        prompt = f"""
        You are a weather expert. Provide the weather details **including the date** in your answer.

        ---
        VERIFIED WEATHER DATA FOR {target_date_str}:
        {final_context}
        ---

        QUESTION: {user_query}

        ANSWER FORMAT:
        - Date: {target_date_str}
        - Max Temp / Min Temp
        - Weather conditions
        - Recommendation (umbrella, jacket, etc.)
        """

        context_for_llm = [final_context]

    ## If the user asked a general question (no specific date) ##
    else:
        context_for_llm = assistant.search_index(user_query)
        if not context_for_llm:
            return "I found no relevant weather information for your general query."

        prompt = f"""
        You are a helpful and concise weather assistant.
        Always include the **date(s)** explicitly in your answer.

        Context:
        ---
        {'\n'.join(context_for_llm)}
        ---

        Question: {user_query}

        Answer:
        - Date(s): state clearly
        - Weather details (Max/Min temp, rain, conditions)
        - Recommendation
        """

    try:
        response = assistant.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API Error: Could not generate response. Details: {e}"

"""## Gradio UI Layout"""

print("🚀 App initialized. Waiting for user to select location and refresh data...")

with gr.Blocks(title="Enhanced OpenWeatherMap Assistant") as demo:
    gr.Markdown(f"""
        # 🌤️ Enhanced RAG Weather Assistant (OpenWeatherMap)
        Ask about the **{FORECAST_DAYS}-day forecast** (Today + 4 Upcoming Days).
    """)

    with gr.Row():
        location_input = gr.Textbox(
            label="1. Enter Location (City, Country OR Landmark OR lat,lon)",
            placeholder="e.g., Hyderabad, IN or 17.3850,78.4867",
            interactive=True,
            scale=3
        )
        refresh_button = gr.Button("Refresh Data", scale=1)
        use_location_button = gr.Button("📍 Use My Location", scale=1)

    status_output = gr.Textbox(
        label="Data Status",
        value=INITIAL_PROMPT,
        interactive=False
    )

    gr.Markdown("## 💬 Ask a Question")

    with gr.Row():
        query_input = gr.Textbox(
            label="Your Weather Question",
            placeholder="e.g., What's the highest temperature tomorrow?",
            lines=2,
            scale=4
        )
        submit_button = gr.Button("Get Answer", scale=1)

    answer_output = gr.Textbox(
        label="Assistant's Answer",
        lines=5,
        interactive=False
    )

    gr.Markdown(f"*(Data provided by OpenWeatherMap, RAG powered by Sentence-Transformers and Gemini-2.5-flash)*")

    location_input.submit(
        fn=update_location_and_index,
        inputs=[location_input],
        outputs=[status_output]
    )
    refresh_button.click(
        fn=update_location_and_index,
        inputs=[location_input],
        outputs=[status_output]
    )

    lat_input = gr.Number(visible=False)
    lon_input = gr.Number(visible=False)

    use_location_button.click(
        fn=use_my_location,
        inputs=[lat_input, lon_input],
        outputs=[status_output],
        js="""
        () => {
            return new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
                    (err) => { alert("Failed to get location: " + err.message); reject(err); }
                );
            });
        }
        """
    )

    submit_button.click(
        fn=answer_weather_query,
        inputs=[query_input],
        outputs=[answer_output]
    )
    query_input.submit(
        fn=answer_weather_query,
        inputs=[query_input],
        outputs=[answer_output]
    )

demo.launch()

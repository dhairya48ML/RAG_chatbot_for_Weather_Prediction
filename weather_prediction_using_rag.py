import os
import requests
import google.generativeai as genai
import gradio as gr
import logging
import numpy as np
from dotenv import load_dotenv # Recommended for local development
import datetime
from collections import Counter

# --- Configuration ---
# Load environment variables from .env file (if present)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeatherRAGAssistant:
    def __init__(self):
        self._load_api_keys()
        self._initialize_gemini()
        self.weather_api_base_url = "http://api.openweathermap.org/data/2.5/weather" # OpenWeatherMap Current Weather
        self.weather_forecast_api_base_url = "http://api.openweathermap.org/data/2.5/forecast" # OpenWeatherMap 5-day Forecast
        self.vector_store = [] # Stores {'text': '...', 'embedding': np.array([...])}
        self.processed_weather_info = [] # Stores processed text chunks before embedding

    def _load_api_keys(self):
        """Loads API keys from environment variables."""
        self.WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        # GOOGLE_API_KEY is mentioned in instructions, but GEMINI_API_KEY covers
        # all Generative AI needs for this specific RAG flow. If another Google
        # service (e.g., Geocoding) were used, GOOGLE_API_KEY would be relevant.

        if not self.WEATHER_API_KEY:
            logging.error("WEATHER_API_KEY not found in environment variables.")
            raise ValueError("WEATHER_API_KEY is required. Please set it in your environment or .env file.")
        if not self.GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is required. Please set it in your environment or .env file.")
        
        logging.info("API keys loaded successfully.")

    def _initialize_gemini(self):
        """Initializes Google Gemini API."""
        try:
            genai.configure(api_key=self.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            self.embedding_model = 'models/embedding-001'
            logging.info("Gemini models initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini models: {e}")
            raise

    def get_weather_data(self, location: str) -> dict:
        """
        Fetches current weather and 5-day forecast data for a given location
        using the OpenWeatherMap API.
        """
        params = {
            "q": location,
            "appid": self.WEATHER_API_KEY,
            "units": "metric" # or "imperial"
        }
        
        weather_data = None
        forecast_data = None

        try:
            # Get current weather
            logging.info(f"Fetching current weather for {location}...")
            weather_response = requests.get(self.weather_api_base_url, params=params)
            weather_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            weather_data = weather_response.json()
            logging.info(f"Successfully fetched current weather for {location}.")

            # Get 5-day forecast
            logging.info(f"Fetching 5-day forecast for {location}...")
            forecast_response = requests.get(self.weather_forecast_api_base_url, params=params)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            logging.info(f"Successfully fetched 5-day forecast for {location}.")

        except requests.exceptions.HTTPError as errh:
            logging.error(f"HTTP Error fetching weather for {location}: {errh}")
            return {"error": f"Failed to retrieve weather data: {errh}"}
        except requests.exceptions.ConnectionError as errc:
            logging.error(f"Connection Error fetching weather for {location}: {errc}")
            return {"error": f"Failed to retrieve weather data: {errc}"}
        except requests.exceptions.Timeout as errt:
            logging.error(f"Timeout Error fetching weather for {location}: {errt}")
            return {"error": f"Failed to retrieve weather data: {errt}"}
        except requests.exceptions.RequestException as err:
            logging.error(f"General Request Error fetching weather for {location}: {err}")
            return {"error": f"Failed to retrieve weather data: {err}"}
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching weather for {location}: {e}")
            return {"error": f"An unexpected error occurred: {e}"}

        return {"current_weather": weather_data, "forecast": forecast_data}

    def process_weather_data(self, weather_api_response: dict) -> str:
        """
        Processes raw weather API response into meaningful text chunks,
        storing them in self.processed_weather_info.
        """
        self.processed_weather_info = [] # Clear previous data

        current_weather = weather_api_response.get("current_weather")
        forecast_data = weather_api_response.get("forecast")
        
        if not current_weather or not forecast_data:
            logging.warning("No complete weather data or forecast data to process.")
            if "error" in weather_api_response:
                return weather_api_response["error"]
            return "Could not retrieve complete weather information."

        city_name = current_weather.get("name", "Unknown City")
        country = current_weather.get("sys", {}).get("country", "Unknown Country")
        
        # Current Weather Chunk
        current_temp = current_weather.get("main", {}).get("temp")
        feels_like = current_weather.get("main", {}).get("feels_like")
        description = current_weather.get("weather", [{}])[0].get("description", "N/A")
        humidity = current_weather.get("main", {}).get("humidity")
        wind_speed = current_weather.get("wind", {}).get("speed")
        sunrise = current_weather.get("sys", {}).get("sunrise")
        sunset = current_weather.get("sys", {}).get("sunset")

        current_weather_text = (
            f"Current weather conditions for {city_name}, {country}:\n"
            f"Temperature: {current_temp}°C (feels like {feels_like}°C)\n"
            f"Conditions: {description.capitalize()}\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s\n"
            f"Sunrise: {self._format_timestamp(sunrise)}\n"
            f"Sunset: {self._format_timestamp(sunset)}"
        )
        self.processed_weather_info.append(current_weather_text)
        logging.debug(f"Processed current weather chunk:\n{current_weather_text}")

        # 5-day Forecast Chunk
        if forecast_data and forecast_data.get("list"):
            forecast_summary = f"Here is the 5-day weather forecast summary for {city_name}, {country}:\n"
            
            daily_forecasts = {}
            for item in forecast_data["list"]:
                dt_txt = item["dt_txt"] # e.g., "2023-10-27 12:00:00"
                date = dt_txt.split(" ")[0]
                if date not in daily_forecasts:
                    daily_forecasts[date] = []
                daily_forecasts[date].append(item)
            
            for date, readings in daily_forecasts.items():
                min_temp = min(r["main"]["temp_min"] for r in readings)
                max_temp = max(r["main"]["temp_max"] for r in readings)
                
                # Get most common description for the day
                descriptions = [r["weather"][0]["description"] for r in readings]
                common_description = Counter(descriptions).most_common(1)[0][0]

                forecast_summary += (
                    f"- On {date}: Min Temp: {min_temp}°C, Max Temp: {max_temp}°C, "
                    f"Conditions: {common_description.capitalize()}.\n"
                )
            self.processed_weather_info.append(forecast_summary)
            logging.debug(f"Processed forecast chunk:\n{forecast_summary}")
        else:
            logging.warning("No forecast data available to process.")
        
        return "Weather data processed successfully." if self.processed_weather_info else "Failed to process any weather data into useful information."

    def _format_timestamp(self, timestamp) -> str:
        """Helper to format Unix timestamp to human-readable time."""
        if timestamp:
            return datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M %Z') # Added %Z for timezone if available
        return "N/A"

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for the given text using the Gemini embedding model."""
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT" # Specify task type for better embeddings
            )
            return np.array(response['embedding'])
        except Exception as e:
            logging.error(f"Failed to generate embedding for text (first 50 chars): '{text[:50]}...'. Error: {e}")
            return np.array([]) # Return empty array if embedding fails

    def create_embeddings(self):
        """
        Generates embeddings for all processed weather information chunks
        and populates the vector_store.
        """
        if not self.processed_weather_info:
            logging.warning("No processed weather information to create embeddings from. Skipping.")
            return

        self.vector_store = [] # Clear previous embeddings
        for i, chunk in enumerate(self.processed_weather_info):
            embedding = self._generate_embedding(chunk)
            if embedding.size > 0: # Check if embedding was successfully generated
                self.vector_store.append({'text': chunk, 'embedding': embedding})
                logging.debug(f"Created embedding for chunk {i+1}.")
            else:
                logging.warning(f"Skipped creating embedding for chunk {i+1} due to embedding generation failure.")
        logging.info(f"Created {len(self.vector_store)} embeddings for weather data chunks.")

    def retrieve_context(self, query_embedding: np.ndarray, top_k: int = 2) -> list[str]:
        """
        Retrieves the most relevant text chunks from the vector store
        based on cosine similarity to the query embedding.
        """
        if not self.vector_store:
            logging.warning("Vector store is empty. No context to retrieve.")
            return []
        if query_embedding.size == 0:
            logging.warning("Query embedding is empty. Cannot retrieve context.")
            return []

        similarities = []
        for item in self.vector_store:
            stored_embedding = item['embedding']
            if stored_embedding.size == 0:
                continue
            
            # Calculate cosine similarity
            # numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
            similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
            similarities.append((similarity, item['text']))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        context_chunks = [text for _, text in similarities[:top_k]]
        logging.debug(f"Retrieved {len(context_chunks)} context chunks with similarities: {similarities[:top_k]}.")
        return context_chunks

    def generate_response(self, user_query: str, location: str) -> str:
        """
        Main method to generate a weather-related response using RAG.
        Orchestrates fetching data, processing, embedding, retrieval, and LLM generation.
        """
        if not user_query or not location:
            return "Please provide both a query and a location to get weather information."

        logging.info(f"Received query: '{user_query}' for location: '{location}'")

        # Step 1: Get weather data
        weather_data_response = self.get_weather_data(location)
        if "error" in weather_data_response:
            return f"Error fetching weather data: {weather_data_response['error']}"

        # Step 2: Process weather data into text chunks
        processing_status = self.process_weather_data(weather_data_response)
        if "Failed" in processing_status or "Could not" in processing_status:
            return processing_status
        if not self.processed_weather_info:
            return "Could not process weather data into useful information for RAG."

        # Step 3: Create embeddings for the processed weather information
        self.create_embeddings()
        if not self.vector_store:
            return "Failed to create embeddings for weather data. Cannot perform RAG."

        # Step 4: Generate embedding for the user's query
        query_embedding = self._generate_embedding(user_query)
        if query_embedding.size == 0:
            return "Failed to generate embedding for your query. Cannot retrieve context."

        # Step 5: Retrieve relevant context
        context_chunks = self.retrieve_context(query_embedding, top_k=2) 
        
        if not context_chunks:
            logging.warning("No relevant context found for the query from the fetched weather data.")
            context_string = "No specific weather data available or directly relevant to your query from the fetched data." 
            # Still provide a base context for Gemini to explain why it can't answer
        else:
            context_string = "\n---\n".join(context_chunks) # Use a clear separator for context chunks
            logging.debug(f"Context used:\n{context_string}")

        # Step 6: Generate a response using Gemini
        prompt = (
            f"You are a helpful weather assistant. Based on the following weather information and the user's question, "
            f"provide a concise and accurate answer. If the provided 'Weather Information' does not contain enough data "
            f"to directly answer the specific question, clearly state that the information isn't available from the "
            f"given data, but still try to give any general relevant insight if possible.\n\n"
            f"Weather Information:\n{context_string}\n\n"
            f"User Question: {user_query}\n\n"
            f"Answer:"
        )

        try:
            logging.info("Sending prompt to Gemini model for response generation...")
            response = self.gemini_model.generate_content(prompt)
            
            if response.candidates and response.candidates[0].content.parts:
                full_response_text = response.candidates[0].content.parts[0].text
                logging.info("Gemini response generated successfully.")
                return full_response_text
            else:
                logging.warning(f"Gemini returned no candidates or content parts. Prompt: {prompt}")
                return "I couldn't generate a response from the weather data. Please try rephrasing your question or check the location."
        except Exception as e:
            logging.error(f"Error generating response from Gemini: {e}")
            return f"I encountered an error trying to generate a response: {e}"

# --- Gradio Interface Setup ---
# Initialize the assistant globally. Gradio handles state across calls.
weather_rag_assistant = None

def init_weather_assistant():
    global weather_rag_assistant
    if weather_rag_assistant is None:
        try:
            weather_rag_assistant = WeatherRAGAssistant()
            logging.info("WeatherRAGAssistant initialized for Gradio.")
        except Exception as e:
            logging.error(f"Failed to initialize WeatherRAGAssistant at startup: {e}")
            return None # Indicate initialization failure
    return weather_rag_assistant

def chat_with_weather_assistant(user_query: str, location: str) -> str:
    """Wrapper function for Gradio interface."""
    assistant = init_weather_assistant()
    if assistant is None:
        return "System initialization failed. Please check API keys and logs."
    
    return assistant.generate_response(user_query, location)

if __name__ == "__main__":
    logging.info("Starting Gradio interface setup...")
    
    # Attempt initial setup to catch errors early
    # This part should be within try-except if it's critical for startup
    try:
        init_weather_assistant() # Call once to try and initialize
    except Exception as e:
        logging.critical(f"Critical error during initial WeatherRAGAssistant setup: {e}")
        # Gradio will still launch, but function calls will report the error

    with gr.Blocks(title="Weather RAG Assistant") as demo:
        gr.Markdown(
            """
            # Advanced Weather RAG Assistant
            Ask questions about current weather and 5-day forecast for any city!
            The assistant uses a Retrieval-Augmented Generation (RAG) approach.
            """
        )
        with gr.Row():
            location_input = gr.Textbox(label="Enter City (e.g., London, New York)", placeholder="e.g., London", interactive=True)
        with gr.Row():
            query_input = gr.Textbox(label="Your Question", placeholder="e.g., What's the temperature like tomorrow? Is it going to rain this week?", lines=2, interactive=True)
        with gr.Row():
            submit_button = gr.Button("Get Weather Info")
        with gr.Row():
            output_text = gr.Textbox(label="Response", lines=7, interactive=False)

        submit_button.click(
            fn=chat_with_weather_assistant,
            inputs=[query_input, location_input],
            outputs=output_text
        )
    
    # Launch Gradio interface
    demo.launch(share=False, debug=True)
    logging.info("Gradio interface launched.")

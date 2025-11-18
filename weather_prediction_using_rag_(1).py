import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import faiss
import numpy as np
from datetime import datetime

# Load environment variables from a .env file
load_dotenv()

class Config:
    """Configuration class for API endpoints, model names, and other constants."""
    OPENWEATHER_API_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    OPENWEATHER_GEO_API_BASE_URL = "http://api.openweathermap.org/geo/1.0/direct"
    GEMINI_MODEL_NAME = "gemini-pro"
    # Note: Use 'models/embedding-001' for genai.embed_content
    EMBEDDING_MODEL_NAME = "models/embedding-001"
    TOP_K_RETRIEVAL = 1 # For simple current weather, we often just need the direct weather info

class WeatherRAGAssistant:
    """
    A RAG-based assistant for weather prediction, leveraging OpenWeatherMap
    for data retrieval and Google Gemini for natural language generation.
    """
    def __init__(self):
        """Initializes API keys, Gemini client, and an empty vector store."""
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not self.openweather_api_key:
            raise ValueError("OPENWEATHER_API_KEY not found in environment variables. Please set it.")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")

        # Configure the genai client with the API key
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_client = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)

        self.vector_store = None
        self.context_texts = [] # Stores original text chunks for retrieval

    def _make_api_request(self, url, params):
        """
        Helper method to make HTTP GET requests and handle common API errors.
        Returns JSON data on success, None on failure.
        """
        try:
            response = requests.get(url, params=params, timeout=10) # Added timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - Response: {response.text}")
            return None
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}. Check your network connection.")
            return None
        except requests.exceptions.Timeout as timeout_err:
            print(f"Request timed out: {timeout_err}. The server took too long to respond.")
            return None
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected request error occurred: {req_err}")
            return None
        except ValueError as json_err:
            print(f"Error decoding JSON response: {json_err} - Raw response: {response.text}")
            return None

    def geocode_location(self, location_name):
        """
        Geocodes a location name to its latitude, longitude, and display name
        using the OpenWeatherMap Geocoding API.
        """
        params = {
            "q": location_name,
            "limit": 1, # Get the top result
            "appid": self.openweather_api_key
        }
        data = self._make_api_request(Config.OPENWEATHER_GEO_API_BASE_URL, params)
        if data and isinstance(data, list) and len(data) > 0:
            # Return lat, lon, and the name returned by the API (which might be more precise)
            return data[0].get("lat"), data[0].get("lon"), data[0].get("name")
        print(f"Could not geocode location: {location_name}")
        return None, None, None

    def fetch_weather_data(self, lat, lon):
        """
        Fetches current weather data for given latitude and longitude
        using the OpenWeatherMap API.
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.openweather_api_key,
            "units": "metric" # Use 'imperial' for Fahrenheit, 'standard' for Kelvin
        }
        data = self._make_api_request(Config.OPENWEATHER_API_BASE_URL, params)
        if not data:
            print(f"Failed to fetch weather data for lat={lat}, lon={lon}")
        return data

    def generate_embedding(self, text):
        """Generates an embedding vector for a given text using Google's embedding model."""
        try:
            # The embed_content function returns a dictionary, we need the 'embedding' key
            result = genai.embed_content(model=Config.EMBEDDING_MODEL_NAME, content=text)
            return np.array(result["embedding"])
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
            return None

    def build_vector_store(self, context_data_list):
        """
        Builds a FAISS vector store from a list of text contexts.
        Each text item is embedded and added to the store.
        """
        self.context_texts = context_data_list
        if not self.context_texts:
            print("No context data provided to build vector store.")
            self.vector_store = None
            return

        embeddings = []
        valid_contexts = []
        for text in self.context_texts:
            embedding = self.generate_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
                valid_contexts.append(text) # Keep track of contexts that successfully got embeddings

        if not embeddings:
            print("No valid embeddings generated for context data. Vector store not built.")
            self.vector_store = None
            return

        self.context_texts = valid_contexts # Update to only include valid contexts
        embeddings_array = np.vstack(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        # Initialize FAISS index (FlatL2 is a simple Euclidean distance index)
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings_array)
        print(f"Vector store built with {len(self.context_texts)} context items.")

    def retrieve_context(self, query, top_k=Config.TOP_K_RETRIEVAL):
        """
        Retrieves the most relevant context chunks from the vector store
        based on a query.
        """
        if not self.vector_store or not self.context_texts:
            print("Vector store not initialized or empty. Cannot retrieve context.")
            return []

        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            print("Failed to generate embedding for the query.")
            return []

        # FAISS expects a 2D array for search, even for a single query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Perform similarity search
        distances, indices = self.vector_store.search(query_embedding, top_k)

        # Retrieve the original text contexts using the indices
        retrieved_contexts = [self.context_texts[i] for i in indices[0] if i < len(self.context_texts)]
        return retrieved_contexts

    def generate_response(self, prompt, context_info):
        """
        Generates a natural language response using the Gemini model,
        conditioned on the provided context information.
        """
        # Construct the prompt with context for the LLM
        full_prompt = f"Based on the following weather information, answer the user's request concisely and informatively. If the context does not contain the answer, state that you cannot answer based on the provided information.\n\nWeather Context:\n{context_info}\n\nUser Request: {prompt}"
        try:
            response = self.gemini_client.generate_content(full_prompt)
            # Access the text directly from the response object
            return response.text
        except genai.exceptions.BlockedPromptException as e:
            print(f"Gemini API Blocked: {e.response.prompt_feedback}")
            return "I apologize, but your request or the generated response was blocked by the safety system."
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return "I apologize, but I couldn't generate a response at this time due to an internal error."

    def get_weather_prediction(self, location, user_query):
        """
        Main method to get weather data for a location and generate a
        RAG-enhanced prediction/response based on a user query.
        """
        print(f"Processing request for '{location}' with query: '{user_query}'")

        # 1. Geocode location
        lat, lon, geo_name = self.geocode_location(location)
        if not lat or not lon:
            return f"Error: Could not find geographic coordinates for '{location}'. Please check the spelling or try a different location."

        print(f"Geocoded '{location}' to '{geo_name}' (Lat: {lat}, Lon: {lon})")

        # 2. Fetch current weather data
        weather_data = self.fetch_weather_data(lat, lon)
        if not weather_data:
            return f"Error: Could not fetch weather data for '{geo_name}'. Please try again later."

        # 3. Format weather data into a readable context string
        # Ensure all keys exist before accessing
        main_data = weather_data.get('main', {})
        weather_desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
        wind_data = weather_data.get('wind', {})
        clouds_data = weather_data.get('clouds', {})
        sys_data = weather_data.get('sys', {})

        sunrise_time = datetime.fromtimestamp(sys_data.get('sunrise', 0)).strftime('%H:%M UTC') if sys_data.get('sunrise') else 'N/A'
        sunset_time = datetime.fromtimestamp(sys_data.get('sunset', 0)).strftime('%H:%M UTC') if sys_data.get('sunset') else 'N/A'

        weather_context_string = (
            f"Current weather information for {geo_name} (Lat: {lat}, Lon: {lon}):\n"
            f"Temperature: {main_data.get('temp', 'N/A')}°C\n"
            f"Feels like: {main_data.get('feels_like', 'N/A')}°C\n"
            f"Min Temperature: {main_data.get('temp_min', 'N/A')}°C\n"
            f"Max Temperature: {main_data.get('temp_max', 'N/A')}°C\n"
            f"Pressure: {main_data.get('pressure', 'N/A')} hPa\n"
            f"Humidity: {main_data.get('humidity', 'N/A')}%\n"
            f"Description: {weather_desc.capitalize()}\n"
            f"Wind Speed: {wind_data.get('speed', 'N/A')} m/s\n"
            f"Cloudiness: {clouds_data.get('all', 'N/A')}%\n"
            f"Sunrise: {sunrise_time}\n"
            f"Sunset: {sunset_time}\n"
            f"Data timestamp: {datetime.fromtimestamp(weather_data.get('dt', 0)).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        print("Generated weather context:\n", weather_context_string)

        # 4. Build vector store with the current weather context
        # For current weather, the single formatted string is our primary context.
        # This step demonstrates the RAG component even if it's a single document.
        self.build_vector_store([weather_context_string])

        # 5. Retrieve most relevant context (in this case, it will be the weather_context_string itself)
        retrieved_contexts = self.retrieve_context(user_query, top_k=Config.TOP_K_RETRIEVAL)
        final_context_for_llm = "\n".join(retrieved_contexts) if retrieved_contexts else weather_context_string

        # 6. Generate response using Gemini model
        print("Final context passed to LLM:\n", final_context_for_llm)
        response_text = self.generate_response(user_query, final_context_for_llm)
        return response_text

# Example Usage
if __name__ == "__main__":
    # Ensure you have a .env file with OPENWEATHER_API_KEY and GEMINI_API_KEY
    # Example .env content:
    # OPENWEATHER_API_KEY="your_openweather_api_key"
    # GEMINI_API_KEY="your_gemini_api_key"

    try:
        assistant = WeatherRAGAssistant()

        print("\n--- Test Case 1: Current weather in London ---")
        location1 = "London"
        query1 = "What is the current temperature and conditions in London?"
        result1 = assistant.get_weather_prediction(location1, query1)
        print(f"\nAssistant's response for '{location1}' and query '{query1}':\n{result1}")

        print("\n--- Test Case 2: Weather details in New York ---")
        location2 = "New York"
        query2 = "Tell me about the wind speed and humidity in New York City."
        result2 = assistant.get_weather_prediction(location2, query2)
        print(f"\nAssistant's response for '{location2}' and query '{query2}':\n{result2}")

        print("\n--- Test Case 3: Invalid Location ---")
        location3 = "asdfghjkl" # Non-existent location
        query3 = "What's the weather like there?"
        result3 = assistant.get_weather_prediction(location3, query3)
        print(f"\nAssistant's response for '{location3}' and query '{query3}':\n{result3}")

        print("\n--- Test Case 4: Specific question about sunrise/sunset ---")
        location4 = "Tokyo"
        query4 = "When is sunrise and sunset in Tokyo today?"
        result4 = assistant.get_weather_prediction(location4, query4)
        print(f"\nAssistant's response for '{location4}' and query '{query4}':\n{result4}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

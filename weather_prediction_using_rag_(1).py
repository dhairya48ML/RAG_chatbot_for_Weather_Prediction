import os
import requests
import google.generativeai as genai
from typing import Dict, Any, Optional

class WeatherRAGAssistant:
    """
    A RAG (Retrieval Augmented Generation) assistant designed to provide weather information
    by fetching real-time data and generating responses using the Gemini model.

    API keys for WeatherAPI and Gemini must be set as environment variables:
    - WEATHER_API_KEY
    - GEMINI_API_KEY
    """

    def __init__(self) -> None:
        """
        Initializes the WeatherRAGAssistant by loading API keys from environment variables
        and configuring the Gemini Generative AI model.
        """
        self.weather_api_key: Optional[str] = os.getenv('WEATHER_API_KEY')
        self.gemini_api_key: Optional[str] = os.getenv('GEMINI_API_KEY')

        if not self.weather_api_key:
            raise ValueError("WEATHER_API_KEY environment variable not set.")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')

        # Base URL for the WeatherAPI
        self.weather_api_base_url: str = "http://api.weatherapi.com/v1"

    def get_weather_data(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Fetches current weather data for a specified location using the WeatherAPI.

        Args:
            location: The city or region for which to retrieve weather data.

        Returns:
            A dictionary containing weather data if the request is successful,
            otherwise None.
        """
        endpoint: str = f"{self.weather_api_base_url}/current.json"
        params: Dict[str, str] = {
            "key": self.weather_api_key,
            "q": location
        }
        try:
            # Set a timeout for the request to prevent indefinite waiting
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error fetching weather for {location}: {e}")
            if response.status_code == 401:
                print("Hint: Check if your WeatherAPI key is valid or correctly set.")
            elif response.status_code == 403:
                print("Hint: WeatherAPI key forbidden. Check permissions or subscription status.")
            elif response.status_code == 400 and "No matching location found" in response.text:
                 print(f"Hint: No matching location found for '{location}'.")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching weather for {location}: {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"Timeout error fetching weather for {location}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"An unexpected request error occurred fetching weather for {location}: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing JSON response from WeatherAPI for {location}: {e}")
            return None

    def generate_response(self, user_query: str, weather_data: Optional[Dict[str, Any]]) -> str:
        """
        Generates a natural language response to a user query, augmented with
        retrieved weather data if available.

        Args:
            user_query: The user's original question about the weather.
            weather_data: A dictionary containing current weather data, or None if not found.

        Returns:
            A string containing the generated response from the Gemini model.
        """
        context_prompt: str = ""
        if weather_data:
            try:
                location_name = weather_data['location']['name']
                country = weather_data['location']['country']
                temp_c = weather_data['current']['temp_c']
                condition = weather_data['current']['condition']['text']
                feels_like_c = weather_data['current']['feelslike_c']
                humidity = weather_data['current']['humidity']
                wind_kph = weather_data['current']['wind_kph']

                context_prompt = (
                    f"Current weather in {location_name}, {country}:
" +
                    f"Temperature: {temp_c}°C (feels like {feels_like_c}°C)\n" +
                    f"Condition: {condition}\n" +
                    f"Humidity: {humidity}%\n" +
                    f"Wind: {wind_kph} km/h\n\n"
                )
            except KeyError as e:
                print(f"Error parsing weather data dictionary: Missing key {e}")
                context_prompt = "Could not fully parse the available real-time weather data.\n\n"
        else:
            context_prompt = "No real-time weather data found for the requested location.\n\n"

        full_prompt: str = (
            "You are a helpful AI assistant that provides weather information. "
            "Use the provided weather data to answer the user's question concisely. "
            "If no specific weather data is provided, state that you couldn't find it "
            "and try to answer based on general knowledge or politely ask for more details.
\n"
            f"Weather Data:\n{context_prompt}"
            f"User Query: {user_query}\n\n"
            "Your Answer:"
        )

        try:
            response = self.gemini_model.generate_content(full_prompt)
            # The .text attribute typically provides the consolidated text output.
            return response.text
        except genai.types.BlockedPromptException as e:
            print(f"Gemini API blocked prompt: {e}")
            return "I'm sorry, I cannot process that query due to safety concerns or policy violations. Please try rephrasing."
        except genai.types.StopCandidateException as e:
            print(f"Gemini API stopped generating a candidate response: {e}")
            return "I'm sorry, I encountered an issue generating a complete response. The generation was stopped prematurely."
        except Exception as e: # Catch other potential errors from genai, e.g., API call errors
            print(f"An unexpected error occurred with the Gemini API: {e}")
            return "I'm sorry, I encountered an issue while trying to generate a response. Please try again later."

    def run(self, location: str, query: str) -> str:
        """
        Executes the RAG process: fetches weather data and generates a response.

        Args:
            location: The geographic location for which to get weather.
            query: The user's natural language query.

        Returns:
            The generated response string.
        """
        weather_data = self.get_weather_data(location)
        response = self.generate_response(query, weather_data)
        return response

if __name__ == "__main__":
    # Example usage:
    # Before running, ensure you have set your API keys as environment variables:
    # export WEATHER_API_KEY="YOUR_WEATHER_API_KEY_HERE"
    # export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

    try:
        assistant = WeatherRAGAssistant()

        print("\n--- Weather RAG Assistant Examples ---\n")

        # Example 1: Get weather for a specific city
        user_location_1 = "London"
        user_query_1 = "What's the weather like in London today?"
        print(f"User Query: {user_query_1}")
        result_1 = assistant.run(user_location_1, user_query_1)
        print(f"Assistant: {result_1}\n")

        # Example 2: Another city
        user_location_2 = "New York"
        user_query_2 = "Tell me about the temperature and conditions in New York."
        print(f"User Query: {user_query_2}")
        result_2 = assistant.run(user_location_2, user_query_2)
        print(f"Assistant: {result_2}\n")

        # Example 3: Non-existent location (will return None for weather_data)
        user_location_3 = "NonExistentCityXYZ123"
        user_query_3 = "How is the weather in NonExistentCityXYZ123?"
        print(f"User Query: {user_query_3}")
        result_3 = assistant.run(user_location_3, user_query_3)
        print(f"Assistant: {result_3}\n")

        # Example 4: A more complex query
        user_location_4 = "Tokyo"
        user_query_4 = "Will I need a jacket in Tokyo this afternoon and what's the humidity?"
        print(f"User Query: {user_query_4}")
        result_4 = assistant.run(user_location_4, user_query_4)
        print(f"Assistant: {result_4}\n")

    except ValueError as ve:
        print(f"Initialization Error: {ve}")
        print("Please ensure WEATHER_API_KEY and GEMINI_API_KEY environment variables are set.")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")

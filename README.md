# 🌦️ Weather Prediction Chatbot with RAG  

A **Retrieval-Augmented Generation (RAG) Chatbot** that provides not just raw weather data, but **actionable, human-like advice**. It combines live weather updates with semantic search and a large language model to help users make better daily decisions — like whether to carry an umbrella, plan outdoor activities, or adjust travel schedules.  

---

## 🚀 Features  
- 🌍 **Live Weather Data** – Retrieves real-time weather conditions using **OpenWeather API**.  
- 🔍 **Semantic Search with FAISS** – Efficiently stores and retrieves relevant weather knowledge.  
- 🤖 **AI-Powered Responses** – Explains forecasts in natural, human-like language using **Gemini 2.5 Flash**.  
- 📊 **Actionable Insights** – Answers practical queries like *“Should I carry an umbrella tomorrow?”*.  
- 💻 **User-Friendly UI** – Interactive chatbot built with **Gradio**.  

---

## 🛠️ Tech Stack  
- **Data Source** – OpenWeather API  
- **Embedding Model** – All-MiniLM-L6-v2  
- **Vector Database** – FAISS  
- **LLM (AI Model)** – Gemini 2.5 Flash  
- **Chatbot UI** – Gradio  

---

## 📄 How It Works  
1. User submits a query (e.g., *“Should I carry an umbrella tomorrow in Hyderabad?”*).  
2. Weather data is fetched in real-time from the **OpenWeather API**.  
3. Query + weather info are embedded using **MiniLM**.  
4. **FAISS** retrieves the most relevant vectors.  
5. **Gemini 2.5 Flash** generates a simple, context-aware recommendation.  
6. Response is displayed instantly in the **Gradio UI**.  

✅ Example:  
*User:* “Should I carry an umbrella tomorrow?”  
*Bot:* “Given the light to moderate rain and significant rainfall (29.9mm), it is recommended to carry an umbrella or a raincoat.”  

---

## 🏗️ Architecture  

The system integrates live weather APIs, embeddings, vector retrieval, and an LLM for contextual reasoning.  

<img width="1450" height="502" alt="image" src="https://github.com/user-attachments/assets/3106c33e-a630-4e30-84e4-a4a41c473f8a" />


---

## 📈 Future Enhancements  
- Multi-day forecast support with detailed advice  
- Personalized alerts (commute, farming, events, travel)  
- Deployment with **FastAPI** or **Streamlit** for web access  
- Integration with voice assistants for hands-free predictions  

---

## 💻 How to Run  
1. Clone the repository:  
   ```bash
   gh repo clone dhairya48ML/RAG_chatbot_for_Weather_Prediction
   cd weather-rag-chatbot
2. Run the code

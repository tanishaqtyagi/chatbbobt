import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
  model_name="gemini-2.5-flash",
  generation_config={
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "response_mime_type": "application/json",
  },
  system_instruction="""You MUST ALWAYS respond with a JSON object containing EXACTLY two keys: `"response"` and `"memory"`."""
)

chat_session = model.start_chat(history=[])

try:
    response = chat_session.send_message("Hello")
    content = response.text
    print("RAW Content from Gemini:", repr(content))
    data = json.loads(content)
    print("Parsed data:", data)
except Exception as e:
    print(f"Error parsing LLM response: {type(e).__name__} - {e}")

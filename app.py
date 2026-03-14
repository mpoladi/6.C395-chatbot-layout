"""
Gradio Web Interface for MIT Course Catalog Chatbot

Run locally:
    python app.py

Access in browser:
    http://localhost:7860
"""

import gradio as gr
from src.chat import Chatbot

def create_chatbot():
    chatbot = Chatbot()

    def chat(message, history):
        return chatbot.get_response(message, history)

    demo = gr.ChatInterface(
        chat,
        title="MIT Course Catalog Assistant",
        description=(
            "Ask me anything about MIT courses! I can help you find classes that match your "
            "requirements (CI-H, HASS, REST), interests, and schedule. "
            "Tip: tell me your major, year, and what requirements you still need — I'll give better recommendations. "
            "Since I run on the free tier, you may occasionally see a 503 error. Just try again in a few seconds."
        ),
        examples=[
            "I'm a Course 6 junior who still needs a CI-H. I'm interested in AI ethics.",
            "What are some good intro biology courses with no prerequisites?",
            "I need a HASS elective about history or philosophy, preferably in the afternoon.",
            "What machine learning courses are available for undergraduates?",
            "Tell me about 6.3900 — what are the prerequisites and when is it offered?",
        ],
    )

    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()

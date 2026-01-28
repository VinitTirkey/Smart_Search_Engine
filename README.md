Smart Search Engine - Vinit Tirkey

Hi, this is my project for my Newcastle University application.

The goal of this project was to solve the 'information overload' problem students face during research. By creating an autonomous agent that filters and verifies search results, I aimed to reduce the time spent on manual data gathering.

What it does:
You type a question, and the backend uses an AI agent to decide where to look for the answer. It can search Google for facts, Reddit for student opinions, or Perplexity for deep research. Then, it summarizes everything into one clean answer with sources.

How I built it:
- The backend is Python using Flask.
- The AI logic handles the decision making (using LangChain).
- I used Bright Data for the web scraping part because my original script kept getting blocked by Google's anti-bot protection.

To run it on your machine:
1. Install the libraries: pip install -r requirements.txt
2. You need to make a .env file and put your API keys in it (Bright Data and OpenAI).
3. Run the server: python app.py

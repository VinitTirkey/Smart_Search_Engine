import os
import time
import requests
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# I'm using LangChain here to handle the "thinking" part of the AI
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- CONFIGURATION ---
load_dotenv() # This pulls the API keys from the hidden .env file

# Set up logging so I can see errors in the terminal if the API fails
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)

# Keys for the Bright Data API
# I needed these because Google kept blocking my standard Python requests
BD_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
BD_ZONE = os.getenv("BRIGHTDATA_SERP_ZONE")
BD_GPT_ID = os.getenv("BRIGHTDATA_GPT_DATASET_ID")
BD_PERPLEXITY_ID = os.getenv("BRIGHTDATA_PERPLEXITY_DATASET_ID")

HEADERS = {
    "Authorization": f"Bearer {BD_API_KEY}",
    "Content-Type": "application/json",
}

# --- HELPER FUNCTIONS ---

def fetch_search_results(query, platform_url):
    """
    I created this wrapper function to avoid rewriting the same 
    request logic for Google, Bing, and Reddit.
    """
    print(f"DEBUG: Searching {platform_url} for query: {query}")
    
    payload = {
        "zone": BD_ZONE,
        "url": f"{platform_url}?q={requests.utils.quote(query)}&brd_json=1",
        "format": "raw",
        "country": "US", # Focusing on US results for better English content
    }
    
    try:
        # Using async=true to prevent the browser from freezing while waiting
        response = requests.post(
            "https://api.brightdata.com/request?async=true",
            headers=HEADERS,
            json=payload
        ).json()
        
        # Parsing the specific JSON structure Bright Data returns
        parsed_results = []
        organic_hits = response.get("organic", [])
        
        if not organic_hits:
            return "No results found."

        for item in organic_hits:
            title = item.get('title', 'No Title')
            link = item.get('link', '#')
            desc = item.get('description', 'No description available')
            parsed_results.append(f"Title: {title}\nLink: {link}\nSnippet: {desc}")
            
        return "\n\n".join(parsed_results)[:10000] # Limiting size to save tokens
        
    except Exception as e:
        return f"Error connecting to search API: {str(e)}"

def trigger_dataset_job(query, dataset_id, target_url):
    """
    Handles the async jobs for GPT and Perplexity.
    We have to poll the 'progress' endpoint until the status is 'ready'.
    """
    print(f"DEBUG: Triggering AI Job for {target_url}")
    payload = [{"url": target_url, "prompt": query}]
    
    trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={dataset_id}&format=json"
    
    # Perplexity gives sources, so I need to ask for them specifically
    if "perplexity" in target_url:
        trigger_url += "&custom_output_fields=answer_text_markdown|sources"
    else:
        trigger_url += "&custom_output_fields=answer_text_markdown"

    try:
        resp = requests.post(trigger_url, headers=HEADERS, json=payload)
        snapshot_id = resp.json().get('snapshot_id')
        
        if not snapshot_id:
            return "Error: Failed to start the job."
        
        # Wait loop (Polling)
        while True:
            status_resp = requests.get(
                f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
                headers=HEADERS
            )
            status = status_resp.json()['status']
            
            if status == 'ready':
                break
            time.sleep(2) # Wait 2 seconds before checking again
            
        # Fetch final data
        data_resp = requests.get(
            f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json",
            headers=HEADERS
        )
        data = data_resp.json()[0]
        
        answer = data.get('answer_text_markdown', 'No answer generated.')
        if "perplexity" in target_url:
            sources = data.get('sources', [])
            return f"{answer}\n\n**Sources:** {sources}"
        return answer

    except Exception as e:
        return f"Dataset Job Failed: {str(e)}"

# --- TOOL DEFINITIONS ---
# The @tool decorator tells LangChain that the AI can use these functions

@tool
def google_tool(query: str):
    """Finds facts using Google Search."""
    return fetch_search_results(query, "https://google.com/search")

@tool
def reddit_tool(query: str):
    """Finds discussions and opinions on Reddit."""
    return fetch_search_results(f"site:reddit.com {query}", "https://google.com/search")

@tool
def perplexity_tool(query: str):
    """Uses Perplexity AI for deep research with citations."""
    return trigger_dataset_job(query, BD_PERPLEXITY_ID, "https://www.perplexity.ai")

# --- AGENT SETUP ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)
available_tools = [google_tool, reddit_tool, perplexity_tool]

# I instructed the agent to specifically cite sources so the user knows it's not hallucinating
smart_search_engine = create_react_agent(
    model=llm,
    tools=available_tools,
    prompt=(
        "You are 'Smart Search Engine', a research assistant for students. "
        "Use Google for facts, Reddit for student opinions, and Perplexity for deep dives. "
        "Always summarize the answer clearly and list your sources at the end."
    )
)

# --- FLASK ROUTES ---

@app.route("/", methods=["GET", "POST"])
def home():
    # Handle the JSON request from the JavaScript frontend
    if request.is_json:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        
        if not user_query:
            return jsonify({"error": "Please enter a valid query."}), 400
            
        try:
            # Pass the user's question to the LangGraph agent
            result = smart_search_engine.invoke({"messages": [("human", user_query)]})
            final_answer = result["messages"][-1].content
            return jsonify({"answer": final_answer})
        except Exception as e:
            logging.error(f"Agent Crash: {e}")
            return jsonify({"error": "Internal Server Error. Check logs."}), 500

    # If it's just a normal page load, show the HTML
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
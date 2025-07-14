"""
Robo: Command-line Chatbot

Requirements:
- Python 3.7+
- pip install spacy wikipedia scikit-learn
- python -m spacy download en_core_web_sm
(Optional for Google search: pip install serpapi and set SERPAPI_KEY)

Usage:
$ python chatbot.py
"""

import random
import re
from datetime import datetime
import sys

try:
    import spacy
except ImportError:
    print("spaCy is not installed. Please run: pip install spacy")
    sys.exit(1)
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    print("spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("scikit-learn is not installed. Please run: pip install scikit-learn")
    sys.exit(1)
try:
    import wikipedia
except ImportError:
    print("wikipedia is not installed. Please run: pip install wikipedia")
    sys.exit(1)

# Optional: Google Search
try:
    from serpapi import GoogleSearch
    import os
    SERPAPI_KEY = os.environ.get("SERPAPI_KEY", None)
    serpapi_available = SERPAPI_KEY is not None
except ImportError:
    serpapi_available = False

MAX_CONTEXT = 5

greetings = ["hello", "hi", "greetings", "sup", "what's up", "hey"]
greeting_responses = ["Hi!", "Hey!", "Hello!", "Hi there!", "Greetings!", "Hey, how can I help you?"]
goodbye_inputs = ["bye", "goodbye", "see you", "exit", "quit"]
goodbye_responses = ["Bye! Take care.", "Goodbye!", "See you soon!", "Exiting. Have a nice day!"]
thanks_inputs = ["thanks", "thank you", "thx"]
thanks_responses = ["You're welcome!", "No problem!", "Glad to help!"]
joke_inputs = ["joke", "make me laugh", "funny"]
joke_responses = [
    "Why did the computer show up at work late? It had a hard drive!",
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why was the math book sad? Because it had too many problems.",
    "Why did the robot go on vacation? To recharge its batteries!"
]
smalltalk_inputs = ["how are you", "what's up", "how's it going", "are you real", "who made you"]
smalltalk_responses = [
    "I'm just code, but I'm here to help!",
    "I'm always ready to chat!",
    "I'm a virtual assistant created to help you.",
    "I was created by a human using Python and spaCy!"
]

faq = {
    "what is your name": "My name is Robo, your chatbot assistant.",
    "who are you": "I am an AI chatbot built with Python and spaCy.",
    "what can you do": "I can answer questions about chatbots and NLP, tell jokes, do math, and more!",
    "how do i build a chatbot": "You can build a chatbot using Python and NLP libraries like spaCy or NLTK.",
    "what is a chatbot": "A chatbot is an AI program that simulates human conversation.",
    "what nlp libraries can i use": "You can use spaCy or NLTK for NLP tasks in Python.",
    "how are you": "I'm just code, but I'm here to help!"
}
corpus = list(faq.keys())

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

def get_intent(user_input):
    cleaned = preprocess(user_input)
    if any(word in cleaned for word in greetings):
        return "greeting"
    if any(word in cleaned for word in goodbye_inputs):
        return "goodbye"
    if any(word in cleaned for word in thanks_inputs):
        return "thanks"
    if any(word in cleaned for word in joke_inputs):
        return "joke"
    if any(word in cleaned for word in smalltalk_inputs):
        return "smalltalk"
    if "google" in user_input.lower() or "search" in user_input.lower():
        return "google"
    if is_math_question(user_input):
        return "math"
    if is_datetime_question(user_input):
        return "datetime"
    if is_wikipedia_question(user_input):
        return "wikipedia"
    return "question"

def get_faq_response(user_input):
    cleaned = preprocess(user_input)
    for question, answer in faq.items():
        if cleaned == preprocess(question):
            return answer
    return None

def similarity_response(user_input):
    cleaned = preprocess(user_input)
    all_questions = [preprocess(q) for q in corpus]
    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(all_questions + [cleaned])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argmax()
    score = vals[0, idx]
    if score > 0.5:
        return faq[corpus[idx]]
    return None

def is_math_question(text):
    return bool(re.search(r"[0-9]+\s*[-+*/^%]\s*[0-9]+", text))

def solve_math(text):
    try:
        expr = re.sub(r"[^0-9.\-+*/()% ]", "", text)
        result = eval(expr, {"__builtins__": None}, {})
        return f"The answer is {result}."
    except Exception:
        return "Sorry, I couldn't solve that math problem."

def is_datetime_question(text):
    keywords = ["time", "date", "day", "month", "year"]
    return any(word in text.lower() for word in keywords)

def get_datetime_response(text):
    now = datetime.now()
    if "time" in text.lower():
        return f"The current time is {now.strftime('%H:%M:%S')}."
    elif "date" in text.lower():
        return f"Today's date is {now.strftime('%Y-%m-%d')}."
    elif "day" in text.lower():
        return f"Today is {now.strftime('%A')}.",
    elif "month" in text.lower():
        return f"The current month is {now.strftime('%B')}.",
    elif "year" in text.lower():
        return f"The current year is {now.strftime('%Y')}.",
    else:
        return f"It's {now.strftime('%A, %Y-%m-%d %H:%M:%S')}."

def is_wikipedia_question(text):
    patterns = [r"who is ", r"what is ", r"tell me about ", r"define ", r"explain "]
    return any(re.search(p, text.lower()) for p in patterns)

def extract_keywords(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    if not keywords:
        keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    return keywords[0] if keywords else text

def wikipedia_response(user_input):
    query = extract_keywords(user_input)
    try:
        page = wikipedia.page(query, auto_suggest=False)
        summary = page.summary.split(". ")[0:2]
        summary = ". ".join(summary) + "."
        url = page.url
        return f"{summary}\nLearn more: {url}\nSource: https://www.wikipedia.org/"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your question is ambiguous. Did you mean: {', '.join(e.options[:5])}?\nSource: https://www.wikipedia.org/"
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find any information on that topic.\nSource: https://www.wikipedia.org/"
    except Exception:
        return "Sorry, I couldn't retrieve information right now.\nSource: https://www.wikipedia.org/"

def google_search_response(query):
    if not serpapi_available:
        return "Google Search is not available. Please install serpapi and set SERPAPI_KEY as an environment variable."
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 3
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    if "organic_results" in results:
        snippets = [res.get("snippet", "") for res in results["organic_results"][:3]]
        return "\n\n".join(snippets)
    else:
        return "Sorry, I couldn't find anything on Google."

def get_chatbot_reply(user_input, context=None):
    if context is None:
        context = []
    intent = get_intent(user_input)
    if intent == "greeting":
        return random.choice(greeting_responses)
    elif intent == "goodbye":
        return random.choice(goodbye_responses)
    elif intent == "thanks":
        return random.choice(thanks_responses)
    elif intent == "joke":
        return random.choice(joke_responses)
    elif intent == "smalltalk":
        return random.choice(smalltalk_responses)
    elif intent == "google":
        return google_search_response(user_input)
    elif intent == "math":
        return solve_math(user_input)
    elif intent == "datetime":
        return get_datetime_response(user_input)
    elif intent == "wikipedia":
        return wikipedia_response(user_input)
    else:
        answer = get_faq_response(user_input)
        if answer:
            return answer
        else:
            sim_answer = similarity_response(user_input)
            if sim_answer:
                return sim_answer
            else:
                return wikipedia_response(user_input)

def chatbot():
    print("ROBO: Hi! I'm Robo")
    context = []
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nROBO: Goodbye!")
            break
        if not user_input.strip():
            continue
        reply = get_chatbot_reply(user_input, context)
        print("ROBO:", reply)
        context.append((user_input, reply))
        if len(context) > MAX_CONTEXT:
            context = context[-MAX_CONTEXT:]
        if get_intent(user_input) == "goodbye":
            break

if __name__ == "__main__":
    chatbot()
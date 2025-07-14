# Robo: Command-line Chatbot

A simple, interactive command-line chatbot built with Python. Robo can answer questions, tell jokes, solve math problems, provide date/time info, search Wikipedia (with links), and more—all using spaCy for natural language processing.

## Features
- Natural language understanding using spaCy
- Wikipedia search with direct links ([Wikipedia](https://www.wikipedia.org/))
- Math problem solving
- Date and time responses
- Jokes, greetings, small talk
- FAQ and similarity-based answers
- Optional: Google Search (via SerpAPI)

## Requirements
- Python 3.7+
- Install dependencies:
  ```bash
  pip install spacy wikipedia scikit-learn
  python -m spacy download en_core_web_sm
  ```
- (Optional for Google search):
  ```bash
  pip install serpapi
  # Set your SerpAPI key as an environment variable:
  # On Windows:
  set SERPAPI_KEY=your_api_key
  # On Linux/macOS:
  export SERPAPI_KEY=your_api_key
  ```

## Usage
Run the chatbot from your terminal:
```bash
python chatbot.py
```

When you start the chatbot, you'll see:
```
ROBO: Hi! I'm Robo
```
Then you can begin chatting immediately.

## How to Use
- Type your questions or requests and press Enter.
- To exit, type `bye`, `exit`, or `quit`.

## Notes
- The chatbot does not print feature instructions or usage tips at startup—just the greeting.
- For Wikipedia answers, a link to the relevant page and a reference to [Wikipedia](https://www.wikipedia.org/) are included.
- Google Search requires a SerpAPI key and the `serpapi` package (optional).

## License
This project is for educational purposes. 
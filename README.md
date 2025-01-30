# Multi-Agent Debate System

A Streamlit-based web application that implements a sophisticated multi-agent debate system for processing complex questions through multiple AI agents. The system utilizes both Groq and Google's Gemini AI models to provide comprehensive, well-reasoned responses.

## Features

- Interactive web interface built with Streamlit
- Multi-agent debate system with four specialized agents:
  - Proposer: Generates initial comprehensive solutions
  - Critic: Analyzes and identifies potential improvements
  - Refiner: Creates improved solutions based on criticism
  - Synthesizer: Combines all insights into final answers
- Dual LLM integration (Groq and Google Gemini)
- Chat history tracking and persistence
- Responsive and user-friendly UI with custom styling
- Structured agent responses with clear formatting

## Prerequisites

- Python 3.x
- Groq API key
- Google Gemini API key

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root with your API keys:

    ```plaintext
    GROQ_API_KEY=your-groq-api-key
    GOOGLE_API_KEY=your-google-api-key
    ```

## Running the Application

Launch the application using Streamlit:

```bash
streamlit run app.py
```

The application will start and automatically open in your default web browser.

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and agent logic
- `requirements.txt`: List of Python dependencies
- `chat_history.json`: Persistent storage for chat history
- `.env`: Configuration file for API keys

## Dependencies

- streamlit==1.31.0
- groq==0.4.2
- google-generativeai==0.3.2
- python-dotenv==1.0.0
- rich==13.7.0

## Usage

1. Enter your question in the input field
2. The system will process your question through multiple agents:
   - The Proposer generates an initial solution
   - The Critic analyzes and provides feedback
   - The Refiner improves the solution
   - The Synthesizer creates a final, comprehensive answer
3. View the chat history to see previous discussions and answers

## Note

Make sure to keep your API keys secure and never commit them to version control.

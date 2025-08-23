import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

st.title("Simple LLM-App with Llama 3.2 🤖")

# Initialize Llama 3.2 (1B parameter version) via Ollama
llm = OllamaLLM(model="llama3.2:1b")

# Prompt form
with st.form("query_form"):
    user_input = st.text_area("Enter your prompt:")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    prompt = PromptTemplate(
        input_variables=["prompt"],
        template="{prompt}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(user_input)
    st.info(response)

    """Streamlit → Creates the web interface (text box, button, and display area).

LangChain → Connects the AI model with your app.

Ollama + Llama 3.2 → The AI engine that processes the user’s input and generates the answer.

User Flow →

User types a prompt in the text box.

Clicks Submit.

The AI model (via LangChain + Ollama) processes the prompt.

The response is shown inside the app.
"""
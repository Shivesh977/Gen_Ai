import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os

# loading from env 
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"


# Prompt template

prompt=ChatPromptTemplate(
    [
        ("system","u are a helpful assistant,Please respond to the user queries"),
        ("user","Question : {question}")
    ]
)

def generate_response(llm,question):
    llm=Ollama(llm)
    parser=StrOutputParser()
    chain=prompt|parser|llm
    response=chain.invoke({'question':question})
    return response 


# Title of the app
st.title("Q&A chatbot")

## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",["mistral"])


## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")


if user_input:
    response=generate_response(user_input,llm)
    st.write(response)
    
else : 
    st.write("Please provide user input ")


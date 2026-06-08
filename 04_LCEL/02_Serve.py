# Same implementation as done in 01_simple_llm_LCEL

# Pip install langserve in terminal
# pip install fastapi
# pip install uvicorn 

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os  
from dotenv import load_dotenv 
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)


# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser=StrOutputParser()

# Create chain 
chain=prompt_template|model|parser  

# App definition 

app=FastAPI(title="LangChain server",
            version="1.0",
            description="simple api server using langchain runnable interface ")

# Adding chain routes 
add_routes(
    app,
    chain,
    path="/chain"
)


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)


# To run the program : in terminal : python 02_Serve.py 
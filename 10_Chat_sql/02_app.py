import streamlit as st
from pathlib import Path # used to work with file and folder paths 
from langchain.agents import create_sql_agent # imports langchains sql agent creator 
from langchain.sql_database import SQLDatabase # allows langchian to communicate with database 
from langchain.agents.agent_types import AgentType # imports diff agent types 
from langchain.callbacks import StreamlitCallbackHandler # imports callback handler for streamlit
from langchain.agents.agent_toolkits import SQLDatabaseToolkit #imports toolkit for SQL databases.
from sqlalchemy import create_engine #Imports SQLAlchemy engine creator. creates connection bw python and database
import sqlite3
from langchain_groq import ChatGroq 

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["Use SQLLite 3 Database - Student.db","Connect to your SQL Database"] # giving user two option to choose from 

selected_opt=st.sidebar.radio(label="Choose the DB which u want to chat",options=radio_opt)  # user selects one DB from the available options

if radio_opt.index(selected_opt)==1: # selecting second option : "Connect to your database "
    db_uri=MYSQL
    # User needs to provide all these info to connect to database 
    mysql_host=st.sidebar.text_input("Provide my sql host") # user enters local host
    mysql_user=st.sidebar.text_input("mysql user ") # mysql user
    mysql_password=st.sidebar.text_input("My sql password ",type="password") # user enters password 
    mysql_db=st.sidebar.text_input("my sql database") # give database name
    
else : 
    db_uri=LOCALDB  # selecting first option 


api_key=st.sidebar.text_input(label="Groq api key ",type="password") # getting api key 

if not db_uri: # if database not selected from given options
    st.info("Please enter the database information and uri")

if not api_key:
    st.info("Please add the groq api key ")
    
## llm modle
llm = ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile")

@st.cache_resource(ttl="2h") # for two hours keep in cache ...store function result in memory for 2 hrs without this creates connection everytime page refresh
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None): # creates db connection
    
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute() # path(__file__) :gives path of project/app.py... : .parent: gives project/student.db ..: .absolute : gives D:/project/student.db full path 
        print(dbfilepath) # prints path in terminal 
        
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True) # connect db to filepath in read only mode
        
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db): # checks all field entered
            st.error("Please provide all MySQL connection details.") # if any field missing 
            st.stop() # sreamlit stops 
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))    # mysql connection 
    
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)
    
    
# Toolkit 
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

        
    

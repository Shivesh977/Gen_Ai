import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


## Set upi the Stramlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")


if not groq_api_key:
    st.info("Please add your Groq APPI key to continue")
    st.stop()

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper() # crates wikipedia search utility
wikipedia_tool=Tool(
    name="Wikipedia", # tool name
    func=wikipedia_wrapper.run, # what function to be executed when function is called
    description="A tool for searching the Internet to find the vatious information on the topics mentioned" # agent reads description and decides 

)


## Initializa the MAth tool

math_chain=LLMMathChain.from_llm(llm=llm) #Creates math chain. eg 2*4 : agent sends it here 
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combinig all tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # reads tool description and choose the tool
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state: # checks whether chat exists
    st.session_state["messages"]=[# creates chat history 
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages: # display old msgs  loop through history
    st.chat_message(msg["role"]).write(msg['content']) 



## LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question}) # adds question to memeory
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")



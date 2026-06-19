# ## RAG Q&A Conversation With PDF Including Chat History
# import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain  
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
# from langchain_community.chat_message_histories import ChatMessageHistory # to store conversation history
# from langchain_core.chat_history import BaseChatMessageHistory  # chat history should follow this structure 
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Messageplaceholder is used to insert chat history insside prompt
# from langchain_groq import ChatGroq
# from langchain_core.runnables.history import RunnableWithMessageHistory # helps combine rag+chat  memory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader # reads pdf
# import os

# from dotenv import load_dotenv
# load_dotenv()

# os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
# embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# # setting up stremlit 
# st.title("Conversational RAG with pdf uploads and chat history") # main title of web page..
# st.write("Upload pdf's and chat with their content") # user give input here..or writes something

# # Input the groq api key 
# api_key=st.text_input("Enter the groq  api key : " ,type="password")


# ## Check if groq api key is provided
# if api_key:
#     llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile")

#     ## chat interface

#     session_id=st.text_input("Session ID",value="default_session")
#     ## statefully manage chat history

#     if 'store' not in st.session_state:
#         st.session_state.store={}

#     uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True) # creates upload button 
#     ## Process uploaded  PDF's
#     if uploaded_files: # user uploaded something 
        
#         documents=[] # empty list to store pdf content 
        
#         for uploaded_file in uploaded_files: # process different pdf's one by one
#             temppdf=f"./temp.pdf" # creates temporary filename ..to save in system such that loader get actual file path
#             with open(temppdf,"wb") as file: # open file in write binary mode
#                 file.write(uploaded_file.getvalue()) # writes pdf data into temp.pdf ...uploaded file is saved locally 
#                 file_name=uploaded_file.name # get original file name

#             loader=PyPDFLoader(temppdf)
#             docs=loader.load() # loads pdf text 
#             documents.extend(docs) # add docs into document 
            

#     # Split and create embeddings for the documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#         splits = text_splitter.split_documents(documents)
#         vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
#         retriever = vectorstore.as_retriever()    

#         contextualize_q_system_prompt=( #rewrite user question using history ..ans is not given 
#             "Given a chat history and the latest user question"
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", contextualize_q_system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]
#             )
        
#         history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt) # combines llm ,retreiver, prompt

#         ## Answer question

#         # Answer question
#         system_prompt = (
#                 "You are an assistant for question-answering tasks. "
#                 "Use the following pieces of retrieved context to answer "
#                 "the question. If you don't know the answer, say that you "
#                 "don't know. Use three sentences maximum and keep the "
#                 "answer concise."
#                 "\n\n"
#                 "{context}"
#             )
#         qa_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]   
#             )
        
#         question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
#         rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

#         def get_session_history(session:str)->BaseChatMessageHistory:

        
#         conversational_rag_chain=RunnableWithMessageHistory(
#             rag_chain,get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer"
#         )

#         user_input = st.text_input("Your question:")
#         if user_input:
#             session_history=get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={
#                     "configurable": {"session_id":session_id}
#                 },  # constructs a key "abc123" in `store`.
#             )
#             st.write(st.session_state.store)
#             st.write("Assistant:", response['answer'])
#             st.write("Chat History:", session_history.messages)
# else:
#     st.warning("Please enter the GRoq API Key")






# Making a rag application with different session ...with history including 

# Importing all libraries 
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory # to store conversation history
from langchain_core.chat_history import BaseChatMessageHistory  # chat history should follow this structure 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Messageplaceholder is used to insert chat history insside prompt
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory # helps combine rag+chat  memory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Loading env 
from dotenv import load_dotenv
load_dotenv()

# Taking hugging face token for embeddings 
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


# Title 
st.title("Conversational RAG with pdf uploads and chat history") # main title of web page..
st.write("Upload pdf's and chat with their content") # user give input here..or writes something

# Taking api key as input from users 
api_key=st.text_input("Please enter api key ",type="password")

# if user entered api key 
if api_key:

    # Making llm 
    llm=ChatGroq(model_name="llama-3.3-70b-versatile",groq_api_key=api_key)
    
    # Session id and session state
    session_id=st.text_input("Session_id : ",value="default_session")
    
    # Session state 
    if 'store' not in st.session_state:
        st.session_state.store={}
        
    # Upload file 
    files=st.file_uploader("Upload the file " , type="pdf",accept_multiple_files=True) # accepting multiple files to be uploaded once 
    
    # Check if file got uploaded or not 
    if files: 
        document=[] # to store content of files
        
        for file in files: 
            temppdf=f"./temp.pdf" # creates temporary filename ..to save in system such that loader get actual file path
            with open(temppdf,"wb") as fil: # open file in write binary mode
                fil.write(file.getvalue()) # writes pdf data into temp.pdf ...uploaded file is saved locally 
        
            loader=PyPDFLoader(temppdf)
            docs=loader.load() # loads pdf text 
            document.extend(docs) # add into documents 
        
        
        # Splitting document
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)  #Each chunk will contain 500 characters maximum.  50 characters repeat between two chunks. to avoid loosing context 
        splits=text_splitter.split_documents(docs) #Splits the documents stored in docs.
        # Embeddings 
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Vector database 
        vectordb=Chroma.from_documents(documents=splits,embedding=embeddings)
        # Creating retreiver 
        retreiver=vectordb.as_retriever()
        
        # Making prompt for question 
        contextualize_q_system_prompt=( #rewrite user question using history ..ans is not given 
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        
        # question retreiver
        history_aware_retriever=create_history_aware_retriever(llm,retreiver,contextualize_q_prompt)
        
        # Answer question prompt 
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )
        
        # create stuff document 
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        # make rag chain 
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Taking input from user
        user_input=st.text_input("Ask question ")
        
        # if user has entered question 
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                {"configurable":{"session_id":session_id}}
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
            st.write("Please give valid api key")





import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from datetime import datetime, timedelta
import phonenumbers
from email_validator import validate_email, EmailNotValidError
import re

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'collecting_info' not in st.session_state:
    st.session_state.collecting_info = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}

def init_llm():
    # Initialize HuggingFace model
    model_name = "google/flan-t5-base"  # You can change this to other models
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text2text-generation",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    return llm

def process_documents(files):
    import tempfile
    import os
    
    documents = []
    for file in files:
        if file.name.endswith('.pdf'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Write the uploaded file content to temporary file
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load the PDF from the temporary file
                loader = PyPDFLoader(tmp_file_path)
                documents.extend(loader.load())
            finally:
                # Clean up the temporary file
                os.unlink(tmp_file_path)
    
    if not documents:
        st.error("Please upload valid PDF documents")
        return None
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

def parse_date(date_string):
    try:
        # Handle relative dates
        if "next" in date_string.lower():
            today = datetime.now()
            days = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
            }
            day = next(k for k in days.keys() if k in date_string.lower())
            current_day = today.weekday()
            target_day = days[day]
            days_ahead = target_day - current_day
            if days_ahead <= 0:
                days_ahead += 7
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
        else:
            # Handle other date formats
            date_obj = datetime.strptime(date_string, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")
    except:
        return None

def validate_phone(phone):
    try:
        phone_number = phonenumbers.parse(phone, "US")
        return phonenumbers.is_valid_number(phone_number)
    except:
        return False

def validate_email_address(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def collect_user_information():
    st.write("Please provide your information:")
    
    name = st.text_input("Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email")
    date = st.text_input("Preferred Date (YYYY-MM-DD or 'Next Monday')")
    
    if st.button("Submit"):
        if not name or not phone or not email or not date:
            st.error("Please fill in all fields")
            return None
            
        if not validate_phone(phone):
            st.error("Please enter a valid phone number")
            return None
            
        if not validate_email_address(email):
            st.error("Please enter a valid email address")
            return None
            
        parsed_date = parse_date(date)
        if not parsed_date:
            st.error("Please enter a valid date")
            return None
            
        return {
            "name": name,
            "phone": phone,
            "email": email,
            "date": parsed_date
        }
    return None

def main():
    st.title("Document Chat Assistant")
    
    # File upload
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
    
    if uploaded_files:
        if not st.session_state.conversation:
            vectorstore = process_documents(uploaded_files)
            llm = init_llm()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
    
        user_question = st.text_input("Ask a question about your documents:")
        
        if user_question:
            # Check if user wants to schedule a call
            if any(keyword in user_question.lower() for keyword in ["call me", "contact me", "schedule", "book"]):
                st.session_state.collecting_info = True
            
            if st.session_state.collecting_info:
                user_info = collect_user_information()
                if user_info:
                    st.session_state.user_info = user_info
                    st.session_state.collecting_info = False
                    st.success("Information collected successfully! We'll contact you soon.")
            else:
                response = st.session_state.conversation({"question": user_question})
                st.write("Assistant:", response['answer'])
                
                # Store chat history
                st.session_state.chat_history.append(("User", user_question))
                st.session_state.chat_history.append(("Assistant", response['answer']))
        
        # Display chat history
        if st.session_state.chat_history:
            st.write("Chat History:")
            for role, text in st.session_state.chat_history:
                st.write(f"{role}: {text}")

if __name__ == "__main__":
    main()
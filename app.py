import streamlit as st
from pipeline import process_query  
from pipeline import process_admin_query  
import os

# Create the folder to save uploaded files

ADMIN_PASSWORD = "admin123"
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Placeholder functions for RAG system
def generate_response(query, files=None):
    # Simulate response generation
    return f"Generated response for the query: '{query}'"

# Streamlit App
st.set_page_config(page_title="Agentic RAG System", layout="wide")

# Add Team 83 Box in Top-Right Corner
st.markdown("""
<div style="
    position: absolute; 
    top: 10px; 
    right: 10px; 
    background-color: #444444; 
    color: white; 
    padding: 15px 30px; 
    font-size: 36px; 
    font-weight: bold; 
    border-radius: 5px; 
    text-align: center;">
    Team 83
</div>
""", unsafe_allow_html=True)

# Title with Vertical Line Separator
st.markdown("""
# Agentic RAG System
<div style="border-top: 2px solid #ccc; margin-top: 10px; margin-bottom: 20px;"></div>
""", unsafe_allow_html=True)

# Main layout
left_col, mid_col, right_col = st.columns([1, 2, 1])

# File Upload Section
with left_col:
    st.header("File Upload")

    # Initialize uploaded files list in session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Upload multiple files
    uploaded_files = st.file_uploader("Upload files to assist retrieval", accept_multiple_files=True)

    # Add newly uploaded files to session state and folder
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Add file to session state if it is not already added
            if uploaded_file.name not in [file.name for file in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append(uploaded_file)
                # Save the file to disk
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

        st.success(f"Files have been uploaded")

    # Remove deleted files from the folder and session state
    current_file_names = [file.name for file in uploaded_files]
    files_to_remove = [file for file in st.session_state.uploaded_files if file.name not in current_file_names]
    
    for file in files_to_remove:
        # Remove the file from the folder
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        if os.path.exists(file_path):
            os.remove(file_path)
        # Remove from session state
        st.session_state.uploaded_files.remove(file)

    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Files:")
        for uploaded_file in st.session_state.uploaded_files:
            st.write(f"- {uploaded_file.name}")


# Admin Query Section
with left_col:
    # Admin authentication section
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        # Admin login form
        with st.form(key="admin_login_form"):
            admin_password = st.text_input("Enter Admin Password:", type="password")
            admin_login_button = st.form_submit_button(label="Login as Admin")

        if admin_login_button:
            if admin_password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True  # Set authenticated state
                st.success("Admin login successful!")
            else:
                st.error("Incorrect password. Please try again.")
    else:
        # Admin query form
        with st.form(key="admin_query_form"):
            admin_query = st.text_input("Enter your admin query:")
            submit_admin_button = st.form_submit_button(label="Send Admin Query")

        if submit_admin_button and admin_query:
            # Process admin query and display the response
            response = process_admin_query(admin_query)

            # Display the admin response in the same way as the assistant's response
            st.markdown(f"""
            <div style="padding: 10px 20px; margin: 10px 0; width: fit-content; max-width: 80%; text-align: left; 
                        font-weight: bold; color: #f1f1f1; background-color: #444444; border-radius: 10px;">
                <strong>Admin Response:</strong> {response}
            </div>
            """, unsafe_allow_html=True)


# Chatbot Section in the Middle Column
with mid_col:
    st.header("Agentic RAG System")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "query" not in st.session_state:
        st.session_state.query = ""

    with st.form(key="chat_form"):
        query = st.text_input("Please write your query:", value=st.session_state.query)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and query:
        st.session_state.messages.insert(0, {"role": "user", "content": query})

        # Process the query using the final.py logic
        response = process_query(query)
        st.session_state.messages.insert(1, {"role": "assistant", "content": response})

    st.markdown("### Conversation")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="background-color: #444444; border-radius: 20px; padding: 10px 20px; margin: 10px 0; 
                        width: fit-content; max-width: 80%; text-align: right; color: white; margin-left: auto;">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 10px 20px; margin: 10px 0; width: fit-content; max-width: 80%; text-align: left; 
                        font-weight: bold; color: #f1f1f1;">
                <strong>Assistant:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)


# Right Column for Instructions
with right_col:
    st.header("Instructions")
    st.markdown("""
    - Upload relevant files to enhance retrieval.
    - Type your query and hit *Send* (or press *Enter*).
    - Review the conversation below.
    """)
    st.image("1.jpg", caption="RAG System", use_column_width=True)

import streamlit as st
from chatWithPdf import create_vector_db, get_qa_chain

st.title("Health Information QA")
question= st.text_input("Ask health related questions: ")

if question:
    chain= get_qa_chain()
    response= chain(question)

    st.header("Answer: ")
    st.write(response["result"])
    
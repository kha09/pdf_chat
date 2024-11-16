import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from htmlTemplates import css, bot_template, user_template
import re

def get_pdf_text(pdf_docs):
    text = ""
    st.session_state.pdf_pages = {}  # Store pages for each PDF
    st.session_state.chunk_sources = {}  # Store mapping of chunks to their sources
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        st.session_state.pdf_pages[pdf.name] = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            pdf_text += page_text
            # Store page text and number
            st.session_state.pdf_pages[pdf.name].append({
                'page_num': page_num + 1,
                'text': page_text,
                'highlights': []  # Store multiple highlights per page
            })
        text += pdf_text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Store the chunks with their source locations
    for chunk in chunks:
        for pdf_name, pages in st.session_state.pdf_pages.items():
            for page in pages:
                if chunk in page['text']:
                    st.session_state.chunk_sources[chunk] = {
                        'pdf_name': pdf_name,
                        'page_num': page['page_num']
                    }
    
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer the question directly and concisely.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def highlight_text_in_page(page_text, source_text):
    """Find and store highlight positions in the page text"""
    # Clean and escape the source text for regex
    source_text = re.escape(source_text.strip())
    # Find all occurrences of the source text in the page
    matches = list(re.finditer(source_text, page_text, re.IGNORECASE))
    return matches


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs before asking questions.")
        return

    # Clear previous highlights
    for pdf_name in st.session_state.pdf_pages:
        for page in st.session_state.pdf_pages[pdf_name]:
            page['highlights'] = []

    # Get relevant documents first
    relevant_docs = st.session_state.retriever.get_relevant_documents(user_question)
    
    # Get the response
    response = st.session_state.conversation.invoke(user_question)
    
    # Add the Q&A to chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.extend([
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": response}
    ])

    # Process each relevant document and add highlights
    for doc in relevant_docs:
        source_text = doc.page_content
        # Find this chunk in our stored sources
        if source_text in st.session_state.chunk_sources:
            source = st.session_state.chunk_sources[source_text]
            pdf_name = source['pdf_name']
            page_num = source['page_num']
            
            # Find the page
            for page in st.session_state.pdf_pages[pdf_name]:
                if page['page_num'] == page_num:
                    # Find and store highlights
                    matches = highlight_text_in_page(page['text'], source_text)
                    if matches:
                        page['highlights'].extend(matches)

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(user_template.replace(
                "{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message["content"]), unsafe_allow_html=True)


def display_pdf_page(pdf_name, page_data):
    """Display a PDF page with highlighted text"""
    st.markdown(f"**Page {page_data['page_num']}**")
    
    text = page_data['text']
    highlights = page_data['highlights']
    
    if highlights:
        # Sort highlights in reverse order to process from end to start
        highlights.sort(key=lambda x: x.start(), reverse=True)
        
        # Apply highlights
        for match in highlights:
            start, end = match.span()
            text = (
                text[:start] +
                f'<span style="background-color: #ffd700; padding: 2px; border-radius: 3px;">{text[start:end]}</span>' +
                text[end:]
            )
    
    # Display the text with any highlights
    st.markdown(f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: white;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []
    if "pdf_pages" not in st.session_state:
        st.session_state.pdf_pages = {}
    if "chunk_sources" not in st.session_state:
        st.session_state.chunk_sources = {}
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # Create two columns
    left_col, right_col = st.columns([2, 1])  # 2:1 ratio

    with left_col:
        st.header("Chat with multiple PDFs :books:")
        if not st.session_state.uploaded_pdfs:
            st.info("👈 Please upload your PDFs in the sidebar to get started!")
            
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

    with right_col:
        st.header("PDF Content")
        if st.session_state.uploaded_pdfs:
            # Add a selectbox to choose which PDF to view
            pdf_names = list(st.session_state.pdf_pages.keys())
            selected_pdf = st.selectbox("Select PDF to view", pdf_names)
            
            # Display pages for selected PDF
            if selected_pdf:
                for page in st.session_state.pdf_pages[selected_pdf]:
                    display_pdf_page(selected_pdf, page)
        else:
            st.write("No PDFs uploaded yet")

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                st.session_state.uploaded_pdfs = pdf_docs  # Store PDFs in session state
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation, st.session_state.retriever = get_conversation_chain(
                        vectorstore)
                st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDFs first!")


if __name__ == '__main__':
    main()

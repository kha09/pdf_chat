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
from difflib import SequenceMatcher
import html

def similar(a, b):
    """Calculate text similarity ratio"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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
            # Split page text into sentences
            sentences = [s.strip() for s in re.split(r'[.!?]+', page_text) if s.strip()]
            
            st.session_state.pdf_pages[pdf.name].append({
                'page_num': page_num + 1,
                'text': page_text,
                'sentences': sentences,
                'highlights': []  # Store multiple highlights per page
            })
            pdf_text += page_text
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
    
    Provide a clear and direct answer using information from the context.
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


def find_similar_sentences(answer, page_data, similarity_threshold=0.6):
    """Find sentences similar to the answer in the page"""
    highlights = []
    answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    
    for answer_sent in answer_sentences:
        for sentence in page_data['sentences']:
            similarity = similar(answer_sent, sentence)
            if similarity > similarity_threshold:
                # Find the position of this sentence in the page text
                start = page_data['text'].lower().find(sentence.lower())
                if start != -1:
                    # Find the actual case-sensitive match
                    actual_text = page_data['text'][start:start + len(sentence)]
                    highlights.append({
                        'start': start,
                        'end': start + len(actual_text),
                        'text': actual_text,
                        'similarity': similarity
                    })
    
    return highlights


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

    # Find and highlight similar sentences in all PDFs
    for pdf_name in st.session_state.pdf_pages:
        for page in st.session_state.pdf_pages[pdf_name]:
            # Find sentences similar to the response
            highlights = find_similar_sentences(response, page)
            if highlights:
                page['highlights'].extend(highlights)
            
            # Also highlight sentences from relevant documents
            for doc in relevant_docs:
                source_highlights = find_similar_sentences(doc.page_content, page)
                if source_highlights:
                    page['highlights'].extend(source_highlights)

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
    
    # Escape the entire text content first
    text = html.escape(page_data['text'])
    highlights = page_data['highlights']
    
    if highlights:
        # Sort highlights by similarity (highest first) and position (from end)
        highlights.sort(key=lambda x: (-x.get('similarity', 0), -x['start']))
        
        # Apply highlights with different intensities based on similarity
        for highlight in highlights:
            start = highlight['start']
            end = highlight['end']
            similarity = highlight.get('similarity', 1.0)
            highlight_text = html.escape(highlight['text'])
            
            # Adjust highlight color intensity based on similarity
            opacity = min(1.0, similarity)
            
            # Create the highlighted version of the text
            highlighted_text = f'<span style="background-color: rgba(255, 215, 0, {opacity}); padding: 2px; border-radius: 3px;">{highlight_text}</span>'
            
            # Replace the text at the correct position
            text = text[:start] + highlighted_text + text[end:]
    
    # Display the text with any highlights
    st.markdown(f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: white; font-family: Arial, sans-serif;">
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

    # Create two columns with adjusted ratio (3:4 instead of 2:1)
    left_col, right_col = st.columns([3, 4])  # Changed ratio to give more space to PDF content

    with left_col:
        st.header("Chat with PDFs :books:")
        if not st.session_state.uploaded_pdfs:
            st.info("👈 Please upload your PDFs in the sidebar to get started!")
            
        user_question = st.text_input("Ask a question:")
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

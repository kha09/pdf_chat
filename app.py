import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
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
import io
import fitz  # PyMuPDF

def create_highlighted_pdf(pdf_bytes, highlights):
    """Create a new PDF with highlights"""
    try:
        # Create a temporary buffer and open the PDF
        buffer = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=buffer, filetype="pdf")
        
        # Track if any highlights were successfully added
        highlights_added = False
        
        # Add highlights to each page if they exist
        if highlights:
            for page_num, page in enumerate(doc):
                page_highlights = highlights.get(page_num + 1, [])
                if page_highlights:
                    # Get the page text and structure
                    page_dict = page.get_text("dict")
                    words = page.get_text_words()  # Get word locations
                    
                    for highlight in page_highlights:
                        text_to_highlight = highlight['text']
                        
                        # Method 1: Direct text search
                        text_instances = page.search_for(text_to_highlight, quads=True)
                        
                        # Method 2: Search by words if direct search fails
                        if not text_instances:
                            text_lower = text_to_highlight.lower()
                            current_rect = None
                            word_matches = []
                            
                            for word in words:
                                word_text = word[4].lower()
                                if word_text in text_lower:
                                    if current_rect is None:
                                        current_rect = fitz.Rect(word[0:4])
                                    else:
                                        current_rect.include_rect(fitz.Rect(word[0:4]))
                                    word_matches.append(word_text)
                                    
                                    # Check if we've found all words
                                    matched_text = " ".join(word_matches)
                                    if matched_text in text_lower:
                                        text_instances.append(current_rect)
                                        current_rect = None
                                        word_matches = []
                        
                        # Method 3: Search in text blocks
                        if not text_instances:
                            for block in page_dict["blocks"]:
                                if "lines" in block:
                                    for line in block["lines"]:
                                        if "spans" in line:
                                            for span in line["spans"]:
                                                if text_to_highlight.lower() in span["text"].lower():
                                                    rect = fitz.Rect(span["bbox"])
                                                    text_instances.append(rect)
                        
                        # Add highlight annotations
                        for inst in text_instances:
                            try:
                                # Create yellow transparent highlight
                                opacity = min(1.0, highlight.get('similarity', 1.0))
                                yellow_color = (1, 1, 0)  # RGB for yellow
                                
                                # Create highlight annotation with explicit save
                                annot = page.add_highlight_annot(inst)
                                annot.set_colors(stroke=yellow_color)
                                annot.set_opacity(opacity)
                                annot.update()
                                
                                # Force annotation save
                                page.apply_redactions()
                                page.clean_contents()
                                highlights_added = True
                            except Exception as e:
                                st.error(f"Error adding highlight: {str(e)}")
                                continue
        
        # Save the PDF with explicit options
        output_buffer = io.BytesIO()
        save_options = {
            'clean': True,
            'deflate': True,
            'garbage': 4,
            'linear': True
        }
        
        # Always save the PDF, whether it has highlights or not
        doc.save(output_buffer, **save_options)
        doc.close()
        buffer.close()
        output_buffer.seek(0)
        return output_buffer
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        # Return original PDF if highlighting fails
        buffer.seek(0)
        return buffer

def get_pdf_text(pdf_docs):
    text = ""
    st.session_state.pdf_pages = {}  # Store pages for each PDF
    st.session_state.chunk_sources = {}  # Store mapping of chunks to their sources
    st.session_state.pdf_highlights = {}  # Store highlights for each PDF
    st.session_state.highlighted_pdfs = {}  # Store highlighted versions
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        st.session_state.pdf_pages[pdf.name] = []
        st.session_state.pdf_highlights[pdf.name] = {}  # Initialize highlights for this PDF
        
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
        
        # Create initial highlighted version
        pdf_content = st.session_state.uploaded_pdfs_content[pdf.name]
        highlighted_pdf = create_highlighted_pdf(pdf_content, {})
        st.session_state.highlighted_pdfs[pdf.name] = highlighted_pdf
    
    return text

def update_highlighted_pdf(pdf_name):
    """Update the highlighted version of a PDF"""
    if pdf_name in st.session_state.uploaded_pdfs_content:
        pdf_bytes = st.session_state.uploaded_pdfs_content[pdf_name]
        highlights = st.session_state.pdf_highlights.get(pdf_name, {})
        highlighted_pdf = create_highlighted_pdf(pdf_bytes, highlights)
        st.session_state.highlighted_pdfs[pdf_name] = highlighted_pdf

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

def similar(a, b):
    """Calculate text similarity ratio"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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
    
    # Add the Q&A to chat history at the beginning
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Prepend new messages to the chat history
    st.session_state.chat_history = [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": response}
    ] + st.session_state.chat_history

    with st.spinner("Creating highlighted PDFs..."):
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
            
            # Create highlighted version immediately
            pdf_bytes = st.session_state.uploaded_pdfs_content[pdf_name]
            highlights = st.session_state.pdf_highlights.get(pdf_name, {})
            highlighted_pdf = create_highlighted_pdf(pdf_bytes, highlights)
            st.session_state.highlighted_pdfs[pdf_name] = highlighted_pdf

    # Create a container for chat history
    chat_container = st.container()
    
    # Display chat history (newest first)
    with chat_container:
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
    
    if page_data['highlights']:
        # Store highlights in session state
        if page_data['page_num'] not in st.session_state.pdf_highlights[pdf_name]:
            st.session_state.pdf_highlights[pdf_name][page_data['page_num']] = []
        
        for highlight in page_data['highlights']:
            st.session_state.pdf_highlights[pdf_name][page_data['page_num']].append({
                'text': highlight['text'],
                'similarity': highlight['similarity']
            })
    
    # Get the original text and highlights
    text = page_data['text']
    highlights = page_data['highlights']
    
    if highlights:
        # Sort highlights by position (from end to start to avoid offset issues)
        highlights.sort(key=lambda x: -x['start'])
        
        # Create a list of text segments
        segments = []
        last_end = len(text)
        
        for highlight in highlights:
            start = highlight['start']
            end = highlight['end']
            similarity = highlight.get('similarity', 1.0)
            opacity = min(1.0, similarity)
            
            # Add the text segment after this highlight
            segments.insert(0, html.escape(text[end:last_end]))
            
            # Add the highlighted segment
            highlight_text = html.escape(text[start:end])
            segments.insert(0, f'<span style="background-color: rgba(255, 215, 0, {opacity}); padding: 2px; border-radius: 3px;">{highlight_text}</span>')
            
            last_end = start
        
        # Add any remaining text at the start
        if last_end > 0:
            segments.insert(0, html.escape(text[0:last_end]))
        
        # Join all segments
        text = ''.join(segments)
    else:
        # If no highlights, just escape the text
        text = html.escape(text)
    
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
    if "uploaded_pdfs_content" not in st.session_state:
        st.session_state.uploaded_pdfs_content = {}
    if "pdf_pages" not in st.session_state:
        st.session_state.pdf_pages = {}
    if "pdf_highlights" not in st.session_state:
        st.session_state.pdf_highlights = {}
    if "chunk_sources" not in st.session_state:
        st.session_state.chunk_sources = {}
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "highlighted_pdfs" not in st.session_state:
        st.session_state.highlighted_pdfs = {}

    # Create two columns with adjusted ratio
    left_col, right_col = st.columns([3, 4])

    with left_col:
        # Create a container for the header with logo and title
        header_container = st.container()
        with header_container:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image("logo.jpg", width=50)
            with col2:
                st.header("Chat with PDFs")
        
        if not st.session_state.uploaded_pdfs:
            st.info("👈 Please upload your PDFs in the sidebar to get started!")
        
        # Move the question input to the top
        user_question = st.text_input("Ask a question:")
        
        # Add some space between input and chat history
        st.markdown("<br>", unsafe_allow_html=True)
        
        if user_question:
            handle_userinput(user_question)

    with right_col:
        st.header("PDF Content")
        if st.session_state.uploaded_pdfs:
            # Add a selectbox to choose which PDF to view
            pdf_names = list(st.session_state.pdf_pages.keys())
            selected_pdf = st.selectbox("Select PDF to view", pdf_names)
            
            if selected_pdf and selected_pdf in st.session_state.highlighted_pdfs:
                try:
                    # Style the download button
                    st.markdown("""
                        <style>
                        div.stDownloadButton > button {
                            width: 100%;
                            background-color: #FF4B4B;
                            color: white;
                            font-weight: bold;
                            padding: 0.5rem;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Show download button for highlighted PDF
                    st.download_button(
                        label="📥 Download Highlighted PDF",
                        data=st.session_state.highlighted_pdfs[selected_pdf],
                        file_name=f"highlighted_{selected_pdf}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space after button
                except Exception as e:
                    st.error(f"Error creating PDF: {str(e)}")
                
                # Display pages for selected PDF
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
                with st.spinner("Processing PDFs..."):
                    # Store PDFs in session state
                    st.session_state.uploaded_pdfs = pdf_docs
                    
                    # Store PDF content in session state
                    for pdf in pdf_docs:
                        pdf_content = pdf.read()
                        st.session_state.uploaded_pdfs_content[pdf.name] = pdf_content
                        pdf.seek(0)  # Reset file pointer for later use
                    
                    # Process PDFs and create initial highlighted versions
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation, st.session_state.retriever = get_conversation_chain(vectorstore)
                
                st.success("PDFs processed and ready!")
            else:
                st.warning("Please upload PDFs first!")


if __name__ == '__main__':
    main()

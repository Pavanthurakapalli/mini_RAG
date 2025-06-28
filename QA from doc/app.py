import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state
qa_chain = None
uploaded_file_name = ""

# Function to process uploaded file
def process_file(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

@app.route("/", methods=["GET", "POST"])
def index():
    global qa_chain, uploaded_file_name
    answer = ""
    question = ""

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                uploaded_file_name = filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                qa_chain = process_file(file_path)
                answer = f"File '{uploaded_file_name}' uploaded successfully. Now ask your question."

        elif "question" in request.form:
            question = request.form.get("question", "")
            if question and qa_chain:
                answer = qa_chain.run(question)
            elif not qa_chain:
                answer = "Please upload a file before asking questions."

    return render_template("index.html", question=question, answer=answer, uploaded_file=uploaded_file_name)

if __name__ == "__main__":
    app.run(debug=True)

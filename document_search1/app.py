from flask import Flask,render_template,request,Response,redirect, url_for;
import os
import chromadb
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
# loading variables from .env file
class MyEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # embed the documents
        
        

        sentences = input
    
        model = SentenceTransformer('/home/shubham/Desktop/hackathon/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)


        # Convert embeddings to a list of lists
        embeddings_as_list = [embedding.tolist() for embedding in embeddings]
        
        return embeddings_as_list
app = Flask(__name__)
CURR_DIR=os.getcwd()
print(CURR_DIR)
CHROMA_PATH = f"{CURR_DIR}/doc_search_embedding"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "doc_search"
# embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='/home/shubham/Desktop/hackathon/all-MiniLM-L6-v2')
embeddings = MyEmbeddingFunction()
# print(embeddings("my name is shubham"))
client = chromadb.PersistentClient(CHROMA_PATH)
# 
try:
  client.create_collection(COLLECTION_NAME,metadata={"hnsw:space": "cosine"},embedding_function=embeddings)
except:
  print("collection already exiat")
collection = client.get_collection(name=COLLECTION_NAME,embedding_function=embeddings)
print(collection.query(query_texts=["what is gross salary?"],where={"project":{"$eq":'itr'}}))
def prepare_docs(pdf_docs,project="None"):
    docs = []
    metadata = []
    content = []
    print(pdf_docs,project)
    for pdf in pdf_docs:

        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"],
            "project": project
        })
    print("Content and metadata are extracted from the documents")
    return content, metadata

def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs

@app.route('/')
def home():
    print("trying to access home")
    return render_template("hello.html")

@app.route("/index/<project>")
def getContent(project):
   print(project)
   result = collection.get(where={"project":{"$eq":project}})
   return f"total count is {result}"
@app.route('/upload', methods=["POST"])
def upload():
    file = request.files['fileToUpload']
    if(os.path.exists(f"{CURR_DIR}/static/uploads/{request.form.get('project')}/{file.filename}")==True):
       return "<script> alert(\"file alredy exists\");window.location='/' </script>"
    if(os.path.exists(f"{CURR_DIR}/static/uploads/{request.form.get('project')}")==False):
       os.mkdir(f"{CURR_DIR}/static/uploads/{request.form.get('project')}")
       
    file.save(f"{CURR_DIR}/static/uploads/{request.form.get('project')}/{file.filename}")
    content,metadata=prepare_docs([f"{CURR_DIR}/static/uploads/{request.form.get('project')}/{file.filename}"],request.form.get("project"))
    text_chunks = get_text_chunks(content,metadata)
    # print([text.metadata['project'] for text in text_chunks])
    collection.add(
       ids=[f"{text_chunks[i].metadata['project']}_{i}" for i in range(len(text_chunks))],
       documents=[text.page_content for text in text_chunks],
       metadatas=[text.metadata for text in text_chunks]
    )
    return redirect("/")
    # return json.dumps(text_chunks) #redirect("/")
    return "<script> alert(\"file saved\") </script>"

if __name__ == "__main__":  
  load_dotenv() 
  print(os.getenv("MY_KEY"))
  app.run(debug = True)


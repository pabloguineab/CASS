import os
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import DirectoryLoader

import pickle

openai_api_key = os.getenv("OPENAI_API_KEY")
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        parsingInstruction = """This is a National Law for a widow's pension."""
        parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=parsingInstruction)
        llama_parse_documents = parser.load_data("./data/llei.pdf")
        
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        parsed_data = llama_parse_documents
    
    return parsed_data

def create_vector_database():
    llama_parse_documents = load_or_parse_data()
    if not llama_parse_documents:
        print("Error: No se parsearon documentos!")
        return
    
    print(f"Ejemplo de texto en el primer documento: {llama_parse_documents[0].text[:500]}")

    with open('data/output.md', 'a', encoding='utf-8') as f:
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')
    
    loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)

    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error al cargar documentos desde el directorio: {str(e)}")
        return
    
    print(f"Documentos cargados desde el directorio: {len(documents)}")
    
    if not documents:
        print("Error: No se cargaron documentos!")
        return

    print(f"Contenido del primer documento: {documents[0].page_content[:500]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print(f"Número de trozos tras dividir los documentos: {len(docs)}")
    if docs:
        print(f"Contenido del primer trozo: {docs[0].page_content[:500]}")
    else:
        print("Error: No se crearon trozos a partir de los documentos!")
        return

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) # Ensure this is the correct updated class
    
    try:
        qdrant = Qdrant.from_documents(
            documents=docs,
            embedding=embeddings,
            url=qdrant_url,
            collection_name="rag_cass",
            api_key=qdrant_api_key,
            force_recreate=True
        )

        print('¡Base de datos vectorial creada con éxito!')
    
    except Exception as e:
        print(f"Error al crear la base de datos vectorial: {str(e)}")

if __name__ == "__main__":
    create_vector_database()


# import required dependencies
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
 # Use the correct import for Qdrant

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  # Usar GPT-4 para chat
from qdrant_client import QdrantClient

import chainlit as cl
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

custom_prompt_template = custom_prompt_template = """Use the following pieces of information to answer the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Make sure to respond in Catalan.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

chat_model = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key)

client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)


def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain



def qa_bot():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_cass",
        embedding=embeddings  # Ensure embeddings are passed correctly
    )

    llm = chat_model
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa

@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hola, benvingut a l'assistent virtual de la CASS per a la resolució de pensions de viduïtat. Com et puc ajudar? Si ho desitges pots adjuntar directament la sol·licitud. "
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)




@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    text_elements = [] 


    await cl.Message(content=answer, elements=text_elements).send()


from io import BytesIO
import chainlit as cl


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)
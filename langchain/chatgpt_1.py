import os
import sys
sys.path.append('../..')

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# import constants

# os.environ["OPENAI_API_KEY"] = constants.APIKEY

# from py3810.myUtils import pickle_dump, pickle_load
path_lumen_docs = '..\langchain\docs\lumen\\docs\\'

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('.env\my_api_key.env')) # read local .env file

SECRET_KEY1 = os.environ.get("SECRET_KEY")
DATABASE_PASSWORD2 = os.environ.get("DATABASE_PASSWORD")
print(f"SECRET_KEY = {SECRET_KEY1}")
print(f"DATABASE_PASSWORD = {DATABASE_PASSWORD2}")

openai.api_key  = os.environ['openai_api']
os.environ["OPENAI_API_KEY"] = openai.api_key



# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = TextLoader("C:\\Users\\ping\\Desktop\\docs_raw_merged_cleaned.txt", encoding = 'utf-8') # Use this line if you only need data.txt
  # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt  
  # loader = DirectoryLoader("data/")  
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  # loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None

from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

import os
os.environ["OPENAI_API_KEY"] = "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"

pdfreader = PdfReader("sample.pdf")

for page in pdfreader.pages:
    text = page.extract_text()
    # print(text)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

new_text = text_splitter.split_text(text)


print(new_text)

##Embeddings
embeddings = OpenAIEmbeddings()
documnet_search = FAISS.from_texts(new_text, embeddings)
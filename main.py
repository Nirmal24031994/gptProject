## Integrate code with openai api

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key





## Initilise the Streamlit Framework

st.title("Currently Looking Generic Chatbot")



## Initilize the LLMS
llm =OpenAI(temperature =0.8)


##Create a Text Box for user input
input_text = st.text_input("Search any Topic");

##Our Own Prompts
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about the celebrity {name} ."

    )

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="title")


if input_text:
    st.write(chain.run(input_text))




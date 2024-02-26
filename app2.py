import streamlit as st
from langchain.prompts import PromptTemplate
import os
# from langchain_community.llms import HuggingFaceEndpoint
from langchain import HuggingFaceHub

gemma = "google/gemma-2b"
mixtral = "mistralai/Mixtral-8x7B-Instruct-v0.1"


huggingface_api = 'add Your Huggingface token'
os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api


def get_model_response(topic, blog_style, no_words,model_name):
    
    if model_name=='gemma':
        llm = HuggingFaceHub(repo_id=gemma,model_kwargs={"temperature":0.01, "max_length":256})
    else:
        llm = HuggingFaceHub(repo_id=mixtral,model_kwargs={"temperature":0.01, "max_length":256})


    #Prompt Template

    nntemplate="""
        Write a compelling and informative blog post on the topic of {topic} using a {blog_style} writing style, with approximately {no_words} words.
    """
    
    

    prompt = PromptTemplate(
        input_variables=['topic','blog_style','no_words'],
        template=nntemplate
    )

    response = llm(prompt.format(topic=topic, blog_style=blog_style,no_words=no_words))
    print(response)

    return response
    



## setting the ui using streamlit
st.set_page_config(
    page_title="AI Blog Generator",
    page_icon="🥸👽🤖👺",
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header('AI Blog Generation Using LLama-2 👽')

input_text = st.text_input("Enter Topic For Generation: ")

col1, col2, col3 =st.columns([5,5,5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('For Whom Blog ?', 
    ('Researchers', 'Scientist','Common People'), index=0)

with col3:
    model_name = st.selectbox('Which Model?', 
    ('gemma','mixtral'), index=0)


submit = st.button("Generate Now")

if submit:
    response = get_model_response(input_text,blog_style,no_words, model_name)
    st.write(response)
    


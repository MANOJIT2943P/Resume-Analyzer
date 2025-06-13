import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile, os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#Initialize model and parser and prompts
model=ChatGoogleGenerativeAI(model='gemini-1.5-pro')
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Extract the following section from the given resume: skills, experience, Education \n {text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="You are a career coach. Based on the given job description and resume, analyze skill gaps and suggest improvements in resume. Then return the imporved resume. \n Job description: {JD} \n Resume: {text}",
    input_variables=['JD','text']
)

#Function for Analyze
def Analysis(resume,jobDescription):
    

    #use docloader to get content of resume
    if resume is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp_file:
            tmp_file.write(resume.read())
            tmp_file_path=tmp_file.name

        loader=PyPDFLoader(tmp_file_path)
        docs=loader.load()

        os.remove(tmp_file_path)

    #First call to extract skills,exp...
    chain1= prompt1 | model | parser
    result=chain1.invoke({'text':docs[0].page_content})

    #Second call to get improvements
    chain2=prompt2 | model | parser 
    result=chain2.invoke({
        'JD':jobDescription,
        'text':docs[0].page_content
    })

    return result

st.title("Resume Analyzer")

#Taking Input
resume=st.file_uploader("Upload Your Resume")
jobDescription=st.text_input("Enter Your Job Description")

if st.button("Analyze"):
    st.write(Analysis(resume,jobDescription))
import streamlit as st
from streamlit_quill import st_quill
from langchain_community.document_loaders import PyPDFLoader
import tempfile, os
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from dotenv import load_dotenv
from docx import Document

load_dotenv()

#Initialize model and parser and prompts


parser1=StrOutputParser()
parser2=JsonOutputParser()

prompt1=PromptTemplate(
    template="Extract the following section from the given resume: skills, experience, Education \n {text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="""You are a career coach. Based on the given job description and resume, analyze skill gaps and suggest improvements in resume. Then return the imporved resume.

    Return your response in JSON format:
    {{
        "skill_gaps":Skill gaps mentions with proper text and bullet poins to make it attractive,
        "suggestions":Improvement suggestions mentions with proper text and bullet poins to make it attractive,
        "improved_resume": updated resume content in plain text but formatted as template style with high ATS score
    }}

    In your response , use plain text with - or numbering for bullet points. Do not use special characters like •, •, *, or emojis. Ensure valid JSON formatting with escaped characters if needed.
    
    \n Job description: {JD} 
    \n Resume: {text}""",
    input_variables=['JD','text'],
)

#Function for Analyze
def Analysis(resume,jobDescription):
    
    model=ChatTogether(
        model='meta-llama/Llama-3-70b-chat-hf',
    )

    #use docloader to get content of resume
    if resume is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp_file:
            tmp_file.write(resume.read())
            tmp_file_path=tmp_file.name

        loader=PyPDFLoader(tmp_file_path)
        docs=loader.load()

        os.remove(tmp_file_path)

    #First call to extract skills,exp...
    chain1= prompt1 | model | parser1
    extracted=chain1.invoke({'text':docs[0].page_content})

    #Second call to get improvements
    chain2=prompt2 | model | parser2
    analysis=chain2.invoke({
        'JD':jobDescription,
        'text':docs[0].page_content
    })

    return analysis

st.title("Resume Analyzer")

#Taking Input
resume=st.file_uploader("Upload Your Resume")
jobDescription=st.text_area(
    label="Enter Your Job Description",
    height=500
)

if st.button("Analyze"):
    result=Analysis(resume,jobDescription)

    tab1,tab2=st.tabs(['Analysis and Suggestion','Sample Resume'])

    with tab1:
        st.subheader('⚠️ Your Resume skill gaps as per Job Description')
        st.write(result["skill_gaps"])

        st.subheader('🤗 Suggestion to improve your Resume')
        st.write(result["suggestions"])
    
    with tab2:
        st.subheader("Want a tailored resume? Click the button below")
        st.write(result["improved_resume"])
        
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile, os
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

#Initialize model and parser and prompts


parser1=StrOutputParser()
parser2=JsonOutputParser()

# prompt1=PromptTemplate(
#     template="Extract the following section from the given resume: skills, experience, Education \n {text}",
#     input_variables=['text']
# )

prompt2=PromptTemplate(
    template="""You are a career coach. Based on the given job description and resume, analyze skill gaps and suggest improvements in resume.

    Return your response in JSON format:
    {{
        "skill_gaps":Skill gaps mentions with proper text and bullet poins to make it attractive,
        "suggestions":Improvement suggestions mentions with proper text and bullet poins to make it attractive,
    }}

    In your response , use plain text with - or numbering for bullet points. Do not use special characters like •, •, *, or emojis. Ensure valid JSON formatting with escaped characters if needed.
    
    \n Job description: {JD} 
    \n Resume: {text}""",
    input_variables=['JD','text'],
)

prompt3=PromptTemplate(
    template="""You are an expert HTML resume template designer.

Generate a clean, semantic, and responsive HTML template for a resume based on an existing resume and given suggestion to improve using the following layout specifications:

Layout Instructions:
- The full name and contact details should be centered at the top.
- Section headers should be in ALL CAPS, bold, and separated by horizontal lines.
- Sections to include (in this order): Education, Work Experience, Activities, University Projects, Additional.
- Each section item should follow this structure:
  • Title (e.g., degree/job title) and Institution/Company aligned to the left  
  • Date aligned to the right  
  • Subheading or role in italic below the title line (optional)  
  • Bullet point list of achievements/responsibilities below  
- Use semantic HTML (<section>, <header>, <ul>, <li>, etc.)
- Apply minimal, clean inline CSS styling:
  • Professional font (e.g., Arial or sans-serif)
  • Adequate spacing between sections
  • Left-aligned body text
  • Use <hr> for section dividers

Constraints:
- Use only HTML and inline CSS (no JavaScript or external CSS)
- Output only the HTML code (no explanations)
- Keep it ready for rendering in a browser

Generate the full HTML resume structure as described above (leave content placeholders where needed).

\n suggestion: {suggestion} 
\n existing Resume: {text}""",
    input_variables=['suggestion','text']
)

model=ChatTogether(
        model='meta-llama/Llama-3-70b-chat-hf',
    )

#Function for Analyze
def Analysis(jobDescription):
    #First call to extract skills,exp...
    # chain1= prompt1 | model | parser1
    # extracted=chain1.invoke({'text':docs[0].page_content})

    #Second call to get improvements
    chain2=prompt2 | model | parser2
    analysis=chain2.invoke({
        'JD':jobDescription,
        'text':docs[0].page_content
    })

    return analysis

def make_resume(suggestion):

    chain3=prompt3 | model | parser1
    improved_resume=chain3.invoke({
        'suggestion':suggestion,
        'text':docs[0].page_content
    })

    return improved_resume

st.title("Resume Analyzer")

#Taking Input
resume=st.file_uploader("Upload Your Resume")
jobDescription=st.text_area(
    label="Enter Your Job Description",
    height=500
)

#use docloader to get content of resume
if resume is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp_file:
        tmp_file.write(resume.read())
        tmp_file_path=tmp_file.name

    loader=PyPDFLoader(tmp_file_path)
    docs=loader.load()

    os.remove(tmp_file_path)

if st.button("Analyze"):
    result=Analysis(jobDescription)
    suggestion=result['skill_gaps']+" "+ result['suggestions']

    improved_resume=make_resume(suggestion)

    tab1,tab2=st.tabs(['Analysis and Suggestion','Sample Resume'])

    with tab1:
        st.subheader('⚠️ Your Resume skill gaps as per Job Description')
        st.write(result["skill_gaps"])

        st.subheader('🤗 Suggestion to improve your Resume')
        st.write(result["suggestions"])
    
    with tab2:
        st.subheader("Your Tailored Resume")
        st.html(improved_resume)
        
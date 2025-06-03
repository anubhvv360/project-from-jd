#!/usr/bin/env python
# coding: utf-8

import streamlit as st
# Set the page title and configuration
st.set_page_config(
    page_title="Resume Project Generator",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import google.generativeai as genai
import re

# Initialize token counting
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0

# Get API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Initialize the Gemini model
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.4,
        max_tokens=2000
    )

# Define the prompt template for job description analysis
job_analysis_template = """
You are an expert career consultant with deep knowledge of various industries and domains. 
Analyze the following job description for a role at {company_name} and extract:
1. The specific industry (e.g., Retail, Healthcare, Technology)
2. The specific domain within that industry (e.g., Data Science in Technology, Supply Chain in Retail)
3. Consider the company's {company_name} specialization and focus areas when determining the industry and domain.
4. Determine the seniority level of the role (Entry-level, Mid-level, Senior, Executive)

Focus on the job requirements, responsibilities, and company information. Ignore general information like DEI statements, benefits, and other standard corporate language that doesn't help determine the specific industry and domain.

If provided, use information about {company_name} to refine your analysis.

Format your response exactly like this:
Industry: [Industry Name]
Domain: [Domain Name]
Seniority: [Seniority Level]

Job Description:
{job_description}
"""

# Define the prompt for project generation
project_generation_template = """
You are an industry expert with deep knowledge of {industry} and specifically {domain}. The job is at {company_name} and the seniority level is {seniority}.

Generate 3 impressive and highly specific professional projects that someone could list on their resume to demonstrate relevant experience for a role in {domain} within the {industry} industry, considering the company profile of {company_name}.

For each project:
1. Create a compelling, specific project heading (not generic)
2. Create 3-4 bullet points that describe the project:
   - First bullet MUST include a quantifiable business impact with specific metrics (use realistic numbers)
   - Remaining bullets should describe the specific actions, methodologies, tools, and processes used
   - Use industry-specific terminology, frameworks, and metrics that would be recognized by hiring managers
   - Include specific company types, product categories, or technical details that show deep domain knowledge
   - Avoid vague or generic statements; be detailed and specific enough to be convincing to industry insiders
   - BOLD key terms, tools, metrics, and industry-specific terminology by surrounding them with ** (e.g., **KPI**)

Analyze the job description below for any specific skills, tools, or frameworks to incorporate:
{job_description}

Your response must be formatted in Markdown EXACTLY as follows:

### [Specific Project Title]
* [Business impact with specific metrics with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]

### Project 2: [Specific Project Title]
* [Business impact with specific metrics with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]

### Project 3: [Specific Project Title]
* [Business impact with specific metrics with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]
* [Specific action/methodology with domain-specific details with **key terms bolded**]

CRITICAL: Ensure that each bullet point is on its own separate line with a proper markdown asterisk (*) at the beginning of each line. Do NOT use the bullet character (•).
"""

# Initialize prompt templates
job_analysis_prompt = PromptTemplate(
    input_variables=["company_name", "job_description"],
    template=job_analysis_template
)

project_generation_prompt = PromptTemplate(
    input_variables=["industry", "domain", "company_name", "job_description", "seniority"],
    template=project_generation_template
)

def analyze_job_description(job_description, company_name):
    llm = get_llm()
    chain = LLMChain(prompt=job_analysis_prompt, llm=llm)
    result = chain.run(job_description=job_description, company_name=company_name)
    
    # Parse the result to extract industry and domain
    industry_match = re.search(r'Industry:\s*(.*?)(?:\n|$)', result)
    domain_match = re.search(r'Domain:\s*(.*?)(?:\n|$)', result)
    seniority_match = re.search(r'Seniority:\s*(.*?)(?:\n|$)', result)
    
    industry = industry_match.group(1).strip() if industry_match else "Unknown"
    domain = domain_match.group(1).strip() if domain_match else "Unknown"
    seniority = seniority_match.group(1).strip() if seniority_match else "Mid-level"
    
    return industry, domain, seniority

def generate_projects(industry, domain, job_description, company_name, seniority):
    llm = get_llm()
    chain = LLMChain(prompt=project_generation_prompt, llm=llm)
    projects = chain.run(
        industry=industry, 
        domain=domain, 
        job_description=job_description,
        company_name=company_name,
        seniority=seniority
    )
    return projects

# Main application
st.title("📄 Resume Project Generator from Job Descriptions")
st.markdown("""
This tool analyzes job descriptions and generates tailored project ideas for your resume. 
Simply paste a job description, enter the company name, and get suggested industry-specific projects that showcase your expertise.
""")

# Company name input
company_name = st.text_input("Enter the company name:", "")

# Input options
input_option = st.radio("Select input method:", ["Paste Job Description", "Upload File"])

job_description = ""
if input_option == "Paste Job Description":
    job_description = st.text_area("Paste the job description here:", height=300)
else:
    uploaded_file = st.file_uploader("Upload job description file (TXT only)", type=["txt"])
    if uploaded_file is not None:
        job_description = uploaded_file.read().decode("utf-8")
        st.text_area("File Content (First 500 chars):", job_description[:500] + "...", height=200)

# Process button
if st.button("Generate Resume Projects") and job_description:
    if not company_name:
        st.warning("Please enter a company name for better results.")
        company_name = "Unknown Company"
    
    with st.spinner("Analyzing job description..."):
        # Get token count for query (approximate)
        query_tokens = len(job_description) // 4
        
        # Analyze the job description
        industry, domain, seniority = analyze_job_description(job_description, company_name)
        
        # Generate project suggestions
        with st.spinner(f"Generating project ideas for {industry} - {domain} - {seniority}..."):
            projects = generate_projects(industry, domain, job_description, company_name, seniority)
            
            # Approximate response tokens
            response_tokens = len(projects) // 4
            
            # Display results
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Job Analysis")
                st.markdown(f"**Company:** {company_name}")
                st.markdown(f"**Industry:** {industry}")
                st.markdown(f"**Domain:** {domain}")
                st.markdown(f"**Seniority:** {seniority}")
            
            with col2:
                st.subheader("Suggested Resume Projects")
                st.markdown(projects)
            
            # Update token counts
            st.session_state.query_tokens += query_tokens
            st.session_state.response_tokens += response_tokens
            st.session_state.tokens_consumed += (query_tokens + response_tokens)

# Download button for the results
if 'projects' in locals():
    result_text = f"""
RESUME PROJECTS FOR {company_name}

INDUSTRY: {industry}
DOMAIN: {domain}
SENIORITY LEVEL: {seniority}

# SUGGESTED PROJECTS
{projects}

"""
    st.download_button(
        label="Download Results",
        data=result_text,
        file_name=f"resume_projects_{company_name.replace(' ', '_')}.txt",
        mime="text/plain",
    )

# Display token usage in sidebar
st.sidebar.title("Usage Statistics")
st.sidebar.write(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
st.sidebar.write(f"Query Tokens: {st.session_state.query_tokens}")
st.sidebar.write(f"Response Tokens: {st.session_state.response_tokens}")

# Tips and guidelines
st.sidebar.title("Tips for Best Results")
st.sidebar.markdown("""
- Enter the exact company name for more targeted project suggestions
- Paste the complete job description or upload a txt file for accurate analysis
- The tool automatically filters out irrelevant information like DEI statements
- More detailed job descriptions yield more tailored project suggestions
- Use the generated projects as inspiration; customize to your experience
""")

# Example of formatted output
st.sidebar.title("Example Output Format")
st.sidebar.markdown("""
### Project Title
* First bullet with **bold keywords** and impact metrics
* Second bullet with **technical skills** used
* Third bullet explaining the **methodology** applied
""")

# Reset token counts only when explicitly requested
if st.sidebar.button("Reset Usage Counters"):
    st.session_state.tokens_consumed = 0
    st.session_state.query_tokens = 0
    st.session_state.response_tokens = 0
    st.sidebar.success("Counters reset successfully!")

# Footer for Credits
st.markdown("""---""")
st.markdown(
    """
    <div style="background: linear-gradient(to right, blue, purple); padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px; color: white;">
        Made with ❤️ by Anubhav Verma<br>
        Please reach out to anubhav.verma360@gmail.com if you encounter any issues.
    </div>
    """, 
    unsafe_allow_html=True
)

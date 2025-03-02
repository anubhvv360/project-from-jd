#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
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

# Initialize state variables for tracking generated content
if 'has_generated_projects' not in st.session_state:
    st.session_state.has_generated_projects = False
if 'has_generated_backstories' not in st.session_state:
    st.session_state.has_generated_backstories = False
if 'has_generated_resources' not in st.session_state:
    st.session_state.has_generated_resources = False
if 'industry' not in st.session_state:
    st.session_state.industry = ""
if 'domain' not in st.session_state:
    st.session_state.domain = ""
if 'seniority' not in st.session_state:
    st.session_state.seniority = ""
if 'projects' not in st.session_state:
    st.session_state.projects = ""
if 'backstories' not in st.session_state:
    st.session_state.backstories = ""
if 'learning_resources' not in st.session_state:
    st.session_state.learning_resources = ""
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'company_name' not in st.session_state:
    st.session_state.company_name = ""

# Get API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Initialize the Gemini model
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=api_key,
        temperature=0.4,
        max_tokens=8000
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

### Project 1: [Specific Project Title]
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

# Define prompt for project backstories
project_backstory_template = """
You are a career coach specializing in interview preparation for the {industry} industry and {domain} domain. 

For the following project descriptions for a {seniority} level position at {company_name}, create detailed backstories that the candidate can use during interviews when questioned about their experience. 

The backstories should:
1. Be appropriate for the {seniority} level role
2. Include specific challenges faced and how they were overcome
3. Provide realistic context about stakeholders, team dynamics, and decision-making processes
4. Include technical details that demonstrate domain expertise

Here are the projects:
{projects}

For each project, provide:

PROJECT BACKSTORY: [2-3 paragraphs with context, challenges, and approach. Each Paragraph not more than 120 words]

KEY INTERVIEW POINTS:
- [Point about your specific role]
- [Point about challenges overcome]
- [Point about collaboration/stakeholders]
- [Point about technical decisions]

POTENTIAL INTERVIEW QUESTIONS AND ANSWERS:
1. Q: [Specific question about the project]
   A: [Suggested answer with specific details]
2. Q: [Specific question about challenges]
   A: [Suggested answer with specific details]
3. Q: [Specific question about outcomes/metrics]
   A: [Suggested answer with specific details]
"""

# Define prompt for learning resources
learning_resources_template = """
You are an expert career coach and learning specialist in the {industry} industry, specifically in the {domain} domain.

Based on the job description and the following projects that were created for a {seniority} level role at {company_name}, create a comprehensive learning guide for someone preparing for this role.

Projects:
{projects}

Job Description:
{job_description}

Create the following sections:

1. KEY TERMINOLOGY EXPLAINED:
   - Identify 8-10 important technical terms, acronyms, or industry jargon from the projects
   - Provide clear, concise definitions (2-3 sentences each)
   - Focus on terms that would be unfamiliar to someone new to this domain

2. CORE INTERVIEW QUESTIONS:
   - List 5-7 technical interview questions specific to this role/domain
   - Include 2-3 behavioral questions relevant to this role
   - Provide brief guidance on how to approach each question (1-2 sentences)

3. RECOMMENDED LEARNING RESOURCES:
   - 3-4 specific online courses from platforms like Coursera, edX, LinkedIn Learning, or Google Skillshop (with specific course names, not just platform names)
   - 2-3 YouTube channels or specific videos relevant to the skills needed

4. INTERVIEW PREPARATION WEBSITES:
   - List 3-5 specific websites that offer preparation resources for this role/industry
   - Briefly explain what each site offers (1 sentence)
   - Include any domain-specific interview preparation resources (e.g., case interview prep for consulting)

Format your response with clear headings and bullet points. Be specific and practical in your recommendations. Focus on quality resources that would genuinely help someone prepare for this role.
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

project_backstory_prompt = PromptTemplate(
    input_variables=["industry", "domain", "company_name", "projects", "seniority"],
    template=project_backstory_template
)

learning_resources_prompt = PromptTemplate(
    input_variables=["industry", "domain", "company_name", "projects", "job_description", "seniority"],
    template=learning_resources_template
)

def analyze_job_description(job_description, company_name):
    llm = get_llm()
    chain = LLMChain(prompt=job_analysis_prompt, llm=llm)
    result = chain.run(job_description=job_description, company_name=company_name)
    
    # Parse the result to extract industry, domain, and seniority
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

def generate_backstories(industry, domain, company_name, projects, seniority):
    llm = get_llm()
    chain = LLMChain(prompt=project_backstory_prompt, llm=llm)
    backstories = chain.run(
        industry=industry,
        domain=domain,
        company_name=company_name,
        projects=projects,
        seniority=seniority
    )
    return backstories

def generate_learning_resources(industry, domain, company_name, projects, job_description, seniority):
    llm = get_llm()
    chain = LLMChain(prompt=learning_resources_prompt, llm=llm)
    resources = chain.run(
        industry=industry,
        domain=domain,
        company_name=company_name,
        projects=projects,
        job_description=job_description,
        seniority=seniority
    )
    return resources

# Set the page title and configuration
st.set_page_config(
    page_title="Resume Project Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main application
st.title("Resume Project Generator from Job Descriptions")
st.markdown("""
This tool analyzes job descriptions and generates tailored project ideas for your resume. 
Simply paste a job description, enter the company name, and get industry-specific projects that showcase your expertise.
""")

# Company name input
company_name_input = st.text_input("Enter the company name:", value=st.session_state.company_name)

# Input options
input_option = st.radio("Select input method:", ["Paste Job Description", "Upload File"])

job_description_input = ""
if input_option == "Paste Job Description":
    job_description_input = st.text_area("Paste the job description here:", height=300, value=st.session_state.job_description)
else:
    uploaded_file = st.file_uploader("Upload job description file (TXT only)", type=["txt"])
    if uploaded_file is not None:
        job_description_input = uploaded_file.read().decode("utf-8")
        st.text_area("File Content (First 500 chars):", job_description_input[:500] + "...", height=200)

# Process button
if st.button("Generate Resume Projects") and job_description_input:
    if not company_name_input:
        st.warning("Please enter a company name for better results.")
        company_name_input = "Unknown Company"
    
    # Save inputs to session state
    st.session_state.company_name = company_name_input
    st.session_state.job_description = job_description_input
    
    with st.spinner("Analyzing job description..."):
        # Get token count for query (approximate)
        query_tokens = len(job_description_input) // 4
        
        # Analyze the job description
        industry, domain, seniority = analyze_job_description(job_description_input, company_name_input)
        
        # Save analysis results to session state
        st.session_state.industry = industry
        st.session_state.domain = domain
        st.session_state.seniority = seniority
        
        # Generate project suggestions
        with st.spinner(f"Generating project ideas for {industry} - {domain}..."):
            projects = generate_projects(industry, domain, job_description_input, company_name_input, seniority)
            st.session_state.projects = projects
            
            # Approximate response tokens
            response_tokens = len(projects) // 4
            
            # Update token counts
            st.session_state.query_tokens += query_tokens
            st.session_state.response_tokens += response_tokens
            st.session_state.tokens_consumed += (query_tokens + response_tokens)
            
            # Mark projects as generated
            st.session_state.has_generated_projects = True
            
            # Reset other generation flags
            st.session_state.has_generated_backstories = False
            st.session_state.has_generated_resources = False
            
            # Force a rerun to show the results and new buttons
            st.experimental_rerun()

# Display results if projects have been generated
if st.session_state.has_generated_projects:
    st.success("Projects Generated Successfully!")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Job Analysis")
        st.markdown(f"**Company:** {st.session_state.company_name}")
        st.markdown(f"**Industry:** {st.session_state.industry}")
        st.markdown(f"**Domain:** {st.session_state.domain}")
        st.markdown(f"**Seniority Level:** {st.session_state.seniority}")
    
    with col2:
        st.subheader("Suggested Resume Projects")
        st.markdown(st.session_state.projects)
    
    # Generate additional content with separate buttons
    col_backstory, col_resources = st.columns(2)
    
    # Project Backstory Button
    with col_backstory:
        if not st.session_state.has_generated_backstories:
            if st.button("Generate Project Backstories"):
                st.warning("⚠️ Generating backstories will utilize additional computational resources. Consider saving your current results first.")
                
                with st.spinner("Creating project backstories..."):
                    backstories = generate_backstories(
                        st.session_state.industry,
                        st.session_state.domain,
                        st.session_state.company_name,
                        st.session_state.projects,
                        st.session_state.seniority
                    )
                    st.session_state.backstories = backstories
                    st.session_state.has_generated_backstories = True
                    
                    # Update token counts
                    response_tokens = len(backstories) // 4
                    st.session_state.response_tokens += response_tokens
                    st.session_state.tokens_consumed += response_tokens
                    
                    # Force a rerun to show the results
                    st.experimental_rerun()
    
    # Learning Resources Button
    with col_resources:
        if not st.session_state.has_generated_resources:
            if st.button("Generate Learning Resources"):
                st.warning("⚠️ Generating learning resources will utilize additional computational resources. Consider saving your current results first.")
                
                with st.spinner("Compiling learning resources..."):
                    learning_resources = generate_learning_resources(
                        st.session_state.industry,
                        st.session_state.domain,
                        st.session_state.company_name,
                        st.session_state.projects,
                        st.session_state.job_description,
                        st.session_state.seniority
                    )
                    st.session_state.learning_resources = learning_resources
                    st.session_state.has_generated_resources = True
                    
                    # Update token counts
                    response_tokens = len(learning_resources) // 4
                    st.session_state.response_tokens += response_tokens
                    st.session_state.tokens_consumed += response_tokens
                    
                    # Force a rerun to show the results
                    st.experimental_rerun()

    # Display Backstories if they have been generated
    if st.session_state.has_generated_backstories:
        st.subheader("Project Backstories")
        
        # Extract project titles for backstory dropdowns
        project_titles = re.findall(r'### Project \d+: (.*?)$', st.session_state.projects, re.MULTILINE)
        if not project_titles:  # Try alternative pattern if first one doesn't match
            project_titles = re.findall(r'### (.*?)$', st.session_state.projects, re.MULTILINE)
        
        # Split backstories by project
        backstory_sections = st.session_state.backstories.split("PROJECT BACKSTORY:")[1:]  # Skip the first empty split
        
        if len(project_titles) == len(backstory_sections):
            for i, (title, backstory) in enumerate(zip(project_titles, backstory_sections)):
                with st.expander(f"📋 Project Backstory: {title}"):
                    st.markdown(f"PROJECT BACKSTORY:{backstory}")
        else:
            # Fallback if parsing failed
            st.markdown(st.session_state.backstories)
    
    # Display Learning Resources if they have been generated
    if st.session_state.has_generated_resources:
        st.subheader("Learning Repository")
        st.markdown(st.session_state.learning_resources)

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

# Button to start over and clear all session state
if st.sidebar.button("Start Over"):
    # Reset all session state values except token counters
    st.session_state.has_generated_projects = False
    st.session_state.has_generated_backstories = False
    st.session_state.has_generated_resources = False
    st.session_state.industry = ""
    st.session_state.domain = ""
    st.session_state.seniority = ""
    st.session_state.projects = ""
    st.session_state.backstories = ""
    st.session_state.learning_resources = ""
    st.session_state.job_description = ""
    st.session_state.company_name = ""
    st.sidebar.success("All generated content cleared. You can start fresh!")
    st.experimental_rerun()

# Footer for Credits
st.markdown("""---""")
st.markdown(
    """
    <div style="background: linear-gradient(to right, blue, purple); padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px; color: white;">
        Made with ❤️ by Anubhav Verma<br>
        Please reach out to anubhav.verma360@gmail.com in case you encounter any issues.
    </div>
    """, 
    unsafe_allow_html=True
)

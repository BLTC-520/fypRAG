import os
import tempfile
import hashlib
import pickle
import pandas as pd
import concurrent.futures
import streamlit as st
import time
from datetime import datetime, timedelta
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from unstructured.partition.pdf import partition_pdf
import json
from typing import Dict, List, Optional, Tuple
import uuid
import re

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "llama3.2"  # Model for embeddings
ANALYSIS_MODEL = "deepseek-r1:8b"  # Model for analysis
DATA_DIR = "data_cache"
VECTOR_DIR = "chroma_db"
LOG_DIR = "logs"
MAX_WORKERS = min(os.cpu_count() or 4, 8)
BATCH_SIZE = 10

# Ensure directories exist
for directory in [DATA_DIR, VECTOR_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Sample citizen profiles
SAMPLE_PROFILES = {
    "ibrahim": {
        "name": "Ibrahim bin Abdullah",
        "location": "Kampung Baru",
        "area_type": "Rural",
        "state": "Kelantan",
        "household_size": 6,
        "monthly_income": 1800,
        "occupation": "Agricultural Worker",
        "age": 45,
        "gender": "Male",
        "education_level": "Primary School",
        "ethnicity": "Malay",
        "dependents": 4,
        "housing_situation": "Owned with mortgage"
    },
    "mei_ling": {
        "name": "Mei Ling Tan",
        "location": "Petaling Jaya",
        "area_type": "Urban",
        "state": "Selangor",
        "household_size": 3,
        "monthly_income": 4500,
        "occupation": "Office Administrator",
        "age": 32,
        "gender": "Female",
        "education_level": "Bachelor's Degree",
        "ethnicity": "Chinese",
        "dependents": 1,
        "housing_situation": "Rented apartment"
    },
    "raj": {
        "name": "Raj Kumar",
        "location": "Senai",
        "area_type": "Semi-urban",
        "state": "Johor",
        "household_size": 4,
        "monthly_income": 2800,
        "occupation": "Factory Worker",
        "age": 38,
        "gender": "Male",
        "education_level": "Secondary School",
        "ethnicity": "Indian",
        "dependents": 2,
        "housing_situation": "Owned house"
    }
}

# Income group scoring function
def get_income_score(income):
    if income < 2500:
        return 30, "B1 (<RM2,500)"
    elif income < 3169:
        return 25, "B2 (RM2,501-RM3,169)"
    elif income < 3969:
        return 20, "B3 (RM3,170-RM3,969)"
    elif income < 4849:
        return 15, "B4 (RM3,970-RM4,849)"
    elif income < 5879:
        return 10, "M1 (RM4,850-RM5,879)"
    elif income < 7099:
        return 7, "M2 (RM5,880-RM7,099)"
    elif income < 8699:
        return 4, "M3 (RM7,110-RM8,699)"
    elif income < 10959:
        return 2, "M4 (RM8,700-RM10,959)"
    elif income < 15039:
        return 1, "T1 (RM10,961-RM15,039)"
    else:
        return 0, "T2 (>RM15,040)"

# Traditional scoring system
def traditional_scoring(profile: Dict) -> Tuple[bool, str]:
    """Score using traditional rules"""
    income_threshold = 2500
    is_eligible = profile["monthly_income"] < income_threshold
    
    explanation = f"""
    Traditional Assessment Results:
    • Monthly Income: RM{profile['monthly_income']}
    • Income Threshold: RM{income_threshold}
    • Eligibility Status: {'ELIGIBLE' if is_eligible else 'NOT ELIGIBLE'}
    
    Note: The traditional assessment is based solely on income threshold criteria.
    """
    
    return is_eligible, explanation

# Modified dynamic_scoring function
def dynamic_scoring(profile: Dict, retriever=None) -> Tuple[float, str, Dict]:
    """Use LLM-RAG for dynamic scoring, integrating B40/M40/T20 groups"""
    # Income grouping using get_income_score
    income_score, income_group = get_income_score(profile["monthly_income"])
    
    # Always use LLM-RAG mode
    prompt = f"""
    Malaysian official household monthly income groups (2019 data):
    - B40: Low-income group (B1: <RM2,500, B2: RM2,501-RM3,169, B3: RM3,170-RM3,969, B4: RM3,970-RM4,849)
    - M40: Middle-income group (M1: RM4,850-RM5,879, M2: RM5,880-RM7,099, M3: RM7,110-RM8,699, M4: RM8,700-RM10,959)
    - T20: High-income group (T1: RM10,961-RM15,039, T2: >RM15,040)

    Please strictly refer to the above groups, especially considering B40 group (B1-B4) as priority for subsidies.
    All scores should be in percentages (0-100%), and the final score should be the weighted average of all factors.

    Applicant Information:
    - Name: {profile['name']}
    - Location: {profile['location']}, {profile['state']}
    - Area Type: {profile['area_type']}
    - Household Size: {profile['household_size']} persons
    - Monthly Income: RM{profile['monthly_income']} (Group: {income_group}, Base Score: {income_score}%)
    - Occupation: {profile['occupation']}
    - Age: {profile['age']}
    - Gender: {profile['gender']}
    - Education Level: {profile['education_level']}
    - Ethnicity: {profile['ethnicity']}
    - Dependents: {profile['dependents']}
    - Housing Situation: {profile['housing_situation']}
    
    You must respond with a valid JSON object in the following format:
    {{
        "total_score": <number between 0 and 100>,
        "breakdown": {{
            "income": {{
                "score": <number between 0 and 100>,
                "weight": <number between 0 and 1>,
                "explanation": "<explanation of income score>"
            }},
            "household": {{
                "score": <number between 0 and 100>,
                "weight": <number between 0 and 1>,
                "explanation": "<explanation of household score>"
            }},
            "location": {{
                "score": <number between 0 and 100>,
                "weight": <number between 0 and 1>,
                "explanation": "<explanation of location score>"
            }},
            "education": {{
                "score": <number between 0 and 100>,
                "weight": <number between 0 and 1>,
                "explanation": "<explanation of education score>"
            }},
            "occupation": {{
                "score": <number between 0 and 100>,
                "weight": <number between 0 and 1>,
                "explanation": "<explanation of occupation score>"
            }}
        }},
        "explanation": "<overall assessment explanation>",
        "eligibility_status": "ELIGIBLE" or "PARTIALLY ELIGIBLE" or "NOT ELIGIBLE"
    }}
    """
    
    try:
        llm = ChatOllama(
            model=ANALYSIS_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        
        response = llm.invoke(prompt)
        raw_response = response.content
        
        # Log the raw response for debugging
        print(f"Raw LLM Response: {raw_response}")
        
        try:
            # Try to extract JSON from the response
            json_str = extract_json_from_response(raw_response)
            if not json_str:
                raise ValueError("No valid JSON found in response")
                
            result = json.loads(json_str)
            
            # Validate required fields
            if not all(key in result for key in ["total_score", "breakdown", "explanation", "eligibility_status"]):
                raise ValueError("Missing required fields in response")
            
            total_score = min(float(result["total_score"]), 100.0)  # Ensure max 100%
            breakdown = result["breakdown"]
            explanation = result["explanation"]
            eligibility_status = result["eligibility_status"]
            
            # Format the explanation with the breakdown
            formatted_explanation = f"""
Dynamic Assessment Results:
• Overall Score: {total_score:.1f}%
• Eligibility Status: {eligibility_status}

Detailed Scoring Breakdown:
"""
            for category, details in breakdown.items():
                formatted_explanation += f"\n{category.title()}:\n"
                if isinstance(details, dict):
                    score = details.get("score", 0)
                    weight = details.get("weight", 0)
                    cat_explanation = details.get("explanation", "")
                    formatted_explanation += f"  • Score: {score:.1f}% (Weight: {weight:.2f})\n"
                    formatted_explanation += f"  • Explanation: {cat_explanation}\n"
            
            return total_score, formatted_explanation, breakdown
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"LLM response parsing failed: {str(e)}")
            print("Falling back to vectordb insights...")
            
            # Fallback to vectordb insights if available
            if retriever:
                try:
                    # Query relevant insights
                    insights = {}
                    for category in ["income", "household", "location", "education", "occupation"]:
                        query = f"What are the key factors and scoring criteria for {category} in Malaysian subsidy eligibility assessment?"
                        answer, _ = query_document(query, retriever)
                        insights[category] = answer
                    
                    # Generate fallback assessment
                    fallback_prompt = f"""
                    Based on the following insights from Malaysian subsidy policy documents:
                    
                    Income Factors:
                    {insights.get('income', 'No specific income insights available.')}
                    
                    Household Factors:
                    {insights.get('household', 'No specific household insights available.')}
                    
                    Location Factors:
                    {insights.get('location', 'No specific location insights available.')}
                    
                    Education Factors:
                    {insights.get('education', 'No specific education insights available.')}
                    
                    Occupation Factors:
                    {insights.get('occupation', 'No specific occupation insights available.')}
                    
                    Please assess the following applicant:
                    {json.dumps(profile, indent=2)}
                    
                    Respond with a JSON object containing:
                    1. total_score (0-100)
                    2. breakdown of scores for each factor
                    3. explanation
                    4. eligibility_status
                    """
                    
                    fallback_response = llm.invoke(fallback_prompt)
                    fallback_json = extract_json_from_response(fallback_response.content)
                    
                    if fallback_json:
                        fallback_result = json.loads(fallback_json)
                        return dynamic_scoring(profile, retriever)  # Recursive call with fallback result
                    
                except Exception as fallback_error:
                    print(f"Fallback assessment failed: {str(fallback_error)}")
            
            # If all fallbacks fail, return error with raw response
            error_msg = f"Failed to parse LLM response: {str(e)}\nRaw response: {raw_response}"
            print(error_msg)
            return 0.0, f"Error: {error_msg}", {}
            
    except Exception as e:
        error_msg = f"Error in LLM communication: {str(e)}"
        print(error_msg)
        return 0.0, f"Error: {error_msg}", {}

def generate_assessment_report(profile: Dict, traditional_result: Tuple[bool, str], 
                             dynamic_result: Tuple[float, str, Dict]) -> str:
    """Generate assessment report"""
    assessment_id = str(uuid.uuid4())
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Get income group information
    income_score, income_group = get_income_score(profile["monthly_income"])
    
    # Parse dynamic assessment result
    dynamic_score = dynamic_result[0]
    dynamic_status = "APPROVED" if dynamic_score >= 70 else "PARTIALLY APPROVED" if dynamic_score >= 50 else "REJECTED"
    
    # Format dynamic assessment breakdown
    breakdown_text = ""
    if isinstance(dynamic_result[2], dict) and dynamic_result[2]:
        breakdown_text = "\nDetailed Scoring Breakdown:\n"
        for category, details in dynamic_result[2].items():
            if isinstance(details, dict):
                # Try to extract score and weight
                score = details.get("score", 0)
                weight = details.get("weight", 0)
                explanation = details.get("explanation", "")
                
                # If no score/weight found, try to parse from value
                if score == 0 and "value" in details:
                    value_str = str(details["value"])
                    # Try to extract score from text like "falls into the 30 points category"
                    score_match = re.search(r'(\d+)\s*points', value_str)
                    if score_match:
                        score = float(score_match.group(1))
                
                breakdown_text += f"\n{category.replace('_', ' ').title()}:\n"
                if score > 0:
                    breakdown_text += f"  • Score: {score:.1f}/100"
                    if weight > 0:
                        breakdown_text += f" (Weight: {weight:.2f})"
                    breakdown_text += "\n"
                if explanation:
                    breakdown_text += f"  • Explanation: {explanation}\n"
                elif "value" in details:
                    breakdown_text += f"  • Details: {details['value']}\n"
    
    # Get overall explanation
    overall_explanation = ""
    if isinstance(dynamic_result[1], str):
        # Extract only the overall assessment part, removing duplicate breakdown
        parts = dynamic_result[1].split("Detailed Scoring Breakdown:")
        overall_explanation = parts[0].strip()
    elif isinstance(dynamic_result[1], dict):
        overall_explanation = dynamic_result[1].get("explanation", "No explanation provided.")
    
    # Calculate total score if not provided
    if dynamic_score == 0 and isinstance(dynamic_result[2], dict):
        total_score = 0
        for category, details in dynamic_result[2].items():
            if isinstance(details, dict):
                score = details.get("score", 0)
                if score == 0 and "value" in details:
                    value_str = str(details["value"])
                    score_match = re.search(r'(\d+)\s*points', value_str)
                    if score_match:
                        score = float(score_match.group(1))
                total_score += score
        dynamic_score = min(total_score, 100.0)  # Cap at 100
    
    report = f"""
MALAYSIAN SUBSIDY ELIGIBILITY ASSESSMENT REPORT
=============================================
Applicant: {profile['name']}
Assessment Date: {current_date}
Reference ID: {assessment_id}

APPLICANT INFORMATION
-------------------
• Name: {profile['name']}
• Location: {profile['location']}, {profile['state']}
• Area Type: {profile['area_type']}
• Household Size: {profile['household_size']} persons
• Monthly Income: RM{profile['monthly_income']} (Group: {income_group})
• Occupation: {profile['occupation']}
• Age: {profile['age']}
• Gender: {profile['gender']}
• Education Level: {profile['education_level']}
• Ethnicity: {profile['ethnicity']}
• Dependents: {profile['dependents']}
• Housing Situation: {profile['housing_situation']}

ASSESSMENT RESULTS
----------------
• Traditional Eligibility: {'ELIGIBLE' if traditional_result[0] else 'NOT ELIGIBLE'}
• Dynamic Assessment Score: {dynamic_score:.1f}/100
• Eligibility Status: {dynamic_status}

{traditional_result[1]}

DYNAMIC ASSESSMENT
----------------
Overall Score: {dynamic_score:.1f}/100
Eligibility Status: {dynamic_status}

{breakdown_text}

Overall Assessment:
{overall_explanation}

This assessment was generated by the Malaysian Subsidy Eligibility System
"""
    return report

def comparison_tab():
    """Subsidy eligibility comparison tab"""
    st.header("Subsidy Eligibility Comparison")
    
    # Select sample profile
    selected_profile = st.selectbox(
        "Select Sample Profile",
        options=list(SAMPLE_PROFILES.keys()),
        format_func=lambda x: SAMPLE_PROFILES[x]["name"]
    )
    
    profile = SAMPLE_PROFILES[selected_profile]
    
    # Display profile details
    with st.expander("View Profile Details"):
        st.json(profile)
    
    if st.button("Assess Eligibility"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traditional Scoring System")
            traditional_result = traditional_scoring(profile)
            st.write(traditional_result[1])
        
        with col2:
            st.subheader("Dynamic Scoring System")
            dynamic_result = dynamic_scoring(profile, st.session_state.get('retriever'))
            st.write(dynamic_result[1])
            
            # Display score progress bar
            st.progress(dynamic_result[0] / 100)
            
            # Display detailed scores
            if dynamic_result[2]:
                st.write("Detailed Scores:")
                for factor, score in dynamic_result[2].items():
                    st.write(f"- {factor}: {score:.1f}")
        
        # Generate and display assessment report
        report = generate_assessment_report(profile, traditional_result, dynamic_result)
        
        st.download_button(
            label="Download Assessment Report",
            data=report,
            file_name=f"subsidy_assessment_{selected_profile}.txt",
            mime="text/plain"
        )
        
        with st.expander("View Complete Assessment Report"):
            st.text(report)
        
        # Explain why dynamic scoring is better
        st.info("""
        ### Why Dynamic Scoring System is Better?
        
        1. **Comprehensive Assessment**: Considers multiple factors, not just income
        2. **Fairness**: Adjusts weights based on actual circumstances
        3. **Transparency**: Provides detailed scoring rationale and policy basis
        4. **Flexibility**: Can dynamically adjust based on policy changes
        5. **Personalization**: Considers applicant's specific situation
        """)

    # --- Admin: Deepseek rubric-based scoring and report ---
    st.markdown("---")
    st.subheader(":lock: Admin: Deepseek-based Scoring with Recommended Rubric")
    admin_profile = st.selectbox("Select applicant profile for rubric-based scoring", list(SAMPLE_PROFILES.keys()), key="admin_profile")
    admin_profile_data = SAMPLE_PROFILES[admin_profile]
    rubric = st.session_state.get('rubric', None)
    if rubric:
        if st.button("Generate Deepseek Assessment Report", key="admin_deepseek_btn"):
            with st.spinner("Scoring with deepseek and recommended rubric..."):
                # Use the same dynamic scoring function as the main assessment
                dynamic_result = dynamic_scoring(admin_profile_data, st.session_state.get('retriever'))
                traditional_result = traditional_scoring(admin_profile_data)
                
                # Generate report using the same format
                report = generate_assessment_report(admin_profile_data, traditional_result, dynamic_result)
                
                # Display the report
                st.text_area("Deepseek Assessment Report", report, height=400)
                
                # Add download button
                st.download_button(
                    "Download Deepseek Report",
                    report,
                    file_name=f"deepseek_assessment_{admin_profile_data['name']}.txt"
                )
                
                # Display detailed scores if available
                if dynamic_result[2]:
                    st.write("Detailed Scores:")
                    for category, details in dynamic_result[2].items():
                        if isinstance(details, dict):
                            score = details.get("score", 0)
                            weight = details.get("weight", 0)
                            explanation = details.get("explanation", "")
                            st.write(f"- {category.title()}: {score:.1f}/100 (Weight: {weight:.2f})")
                            st.write(f"  Explanation: {explanation}")
    else:
        st.warning("No recommended scoring rubric available. Please run an analysis and generate a rubric first.")

# ===== Core functionality functions =====

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def save_file_metadata(file_hash, file_name, metadata):
    file_metadata = {
        "file_name": file_name,
        "metadata": metadata
    }
    with open(os.path.join(DATA_DIR, f"{file_hash}.pkl"), "wb") as f:
        pickle.dump(file_metadata, f)

def load_file_metadata(file_hash):
    try:
        with open(os.path.join(DATA_DIR, f"{file_hash}.pkl"), "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None

def save_extraction_results(file_hash, tables, text_content):
    results = {
        "tables": tables,
        "text_content": text_content
    }
    with open(os.path.join(DATA_DIR, f"{file_hash}_content.pkl"), "wb") as f:
        pickle.dump(results, f)

def load_extraction_results(file_hash):
    try:
        with open(os.path.join(DATA_DIR, f"{file_hash}_content.pkl"), "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None

def extract_from_pdf(file_path, file_hash=None):
    if file_hash:
        cached_results = load_extraction_results(file_hash)
        if cached_results:
            return cached_results["tables"], cached_results["text_content"]
    
    elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        strategy="hi_res",
        model_name="yolox"
    )
    
    tables = []
    text_content = []
    
    for element in elements:
        if "Table" in str(type(element)):
            tables.append({
                'html': element.metadata.text_as_html if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html') else None,
                'text': str(element)
            })
        else:
            text_content.append(str(element))
    
    text_result = " ".join(text_content)
    
    if file_hash and text_result:
        save_extraction_results(file_hash, tables, text_result)
    
    return tables, text_result

def process_file_pure(uploaded_file_bytes, uploaded_file_name):
    try:
        file_content = uploaded_file_bytes
        file_hash = get_file_hash(file_content)
        save_file_metadata(file_hash, uploaded_file_name, {
            "size": len(file_content),
            "processed_date": pd.Timestamp.now().isoformat()
        })
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            file_path = tmp_file.name
            
        tables, text_content = extract_from_pdf(file_path, file_hash)
        
        try:
            os.unlink(file_path)
        except Exception:
            pass
            
        return {
            "file_name": uploaded_file_name,
            "file_hash": file_hash,
            "tables": tables,
            "text_content": text_content,
            "error": None
        }
    except Exception as e:
        return {
            "file_name": uploaded_file_name,
            "error": str(e),
            "tables": [],
            "text_content": "",
            "file_hash": None
        }

def batch_process_files(uploaded_files):
    results = []
    file_hashes = []
    all_tables = []
    all_text_contents = []

    if not uploaded_files:
        return [], [], []

    status_container = st.empty()
    progress_bar = st.progress(0)
    progress_text = st.empty()
    file_statuses = [{
        "name": f.name,
        "status": "Pending",
        "error": None
    } for f in uploaded_files]

    def update_status_display():
        status_text = ""
        for fs in file_statuses:
            emoji = "✅" if fs["status"] == "Completed" else "⏳" if fs["status"] == "Processing" else "❌" if fs["status"] == "Error" else "🔄"
            status_line = f"{emoji} {fs['name']}: {fs['status']}"
            if fs["error"]:
                status_line += f" - {fs['error']}"
            status_text += status_line + "\n"
        status_container.text(status_text)
        completed = sum(1 for fs in file_statuses if fs["status"] in ["Completed", "Error"])
        progress = completed / len(file_statuses)
        progress_bar.progress(progress)
        progress_text.text(f"Processing progress: {progress*100:.1f}%")

    update_status_display()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, uploaded_file in enumerate(uploaded_files):
            file_statuses[idx]["status"] = "Processing"
            update_status_display()
            futures.append(
                executor.submit(
                    process_file_pure,
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
            )

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            for fs in file_statuses:
                if fs["name"] == result["file_name"]:
                    if result["error"]:
                        fs["status"] = "Error"
                        fs["error"] = result["error"]
                    else:
                        fs["status"] = "Completed"
                    break
            update_status_display()
            if not result["error"] and result.get("text_content"):
                file_hashes.append(result["file_hash"])
                all_tables.extend(result["tables"])
                all_text_contents.append(result["text_content"])

    update_status_display()
    time.sleep(1)
    progress_bar.empty()
    progress_text.empty()
    status_container.empty()

    return file_hashes, all_tables, all_text_contents

def get_or_create_vectordb(contents, file_hashes):
    if not file_hashes:
        st.error("No file hashes provided")
        return None
        
    collection_id = hashlib.md5(("_".join(sorted(file_hashes))).encode()).hexdigest()
    collection_name = f"pdf_content_{collection_id}"
    
    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL
        )
        
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=VECTOR_DIR
        )
        
        # Check if vector database already exists
        if vectordb._collection.count() > 0:
            st.success(f"Using existing vector database with {vectordb._collection.count()} entries")
            return vectordb.as_retriever(search_kwargs={"k": 8})
            
        # If there's no content but there are file hashes, try to load from cache
        if not contents and file_hashes:
            st.info("Attempting to load content from cache...")
            cached_contents = []
            for file_hash in file_hashes:
                cached_result = load_extraction_results(file_hash)
                if cached_result and cached_result.get("text_content"):
                    cached_contents.append(cached_result["text_content"])
            
            if cached_contents:
                contents = cached_contents
                st.success(f"Successfully loaded {len(cached_contents)} documents from cache")
            else:
                st.error("No cached content found for the provided file hashes")
                return None
            
        if not contents:
            st.error("No content available to create vector database")
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        chunks = []
        for content in contents:
            chunks.extend(text_splitter.create_documents([content]))
        
        if not chunks:
            st.error("No text chunks could be created from the documents")
            return None
            
        st.info(f"Creating new vector database with {len(chunks)} chunks")
        
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=VECTOR_DIR
        )
        return vectordb.as_retriever(search_kwargs={"k": 8})
        
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        return None

# --- Custom Query and Key Insights use llama3.2, English only, low temperature, cite sources ---
def query_document(question, retriever):
    if not retriever:
        return "Error: Retriever not available", []
    template = '''
    You are a policy analysis expert. Based only on the following context from research documents, answer the question in English.
    For every key point in your answer, cite the most relevant sentence verbatim as [Source n].
    If you cannot find an answer, say "No relevant information found."
    Context:
    {context}
    Question: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(
        model="llama3.2",
        base_url=OLLAMA_BASE_URL,
        temperature=0.05,
        timeout=60
    )
    relevant_docs = retriever.get_relevant_documents(question)
    if not relevant_docs:
        return "No relevant information found.", []
    context = "\n\n".join([f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(relevant_docs)])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, relevant_docs

# --- Key Insights/Analysis Questions use llama3.2, English only, low temperature ---
def parallel_query_documents(questions, retriever):
    if not retriever:
        return {}
    results = {}
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_questions = len(questions)
    completed = [0]
    def update_progress():
        progress_val = min(completed[0] / total_questions, 1.0)
        progress_bar.progress(progress_val)
        progress_text.text(f"Analysis progress: {progress_val*100:.1f}%")
    update_progress()
    try:
        def worker_query(question_item):
            question_text = question_item["question"]
            question_key = question_item["key"]
            try:
                # Use llama3.2 for all key insights
                answer, relevant_docs = query_document(question_text, retriever)
                completed[0] += 1
                return question_key, answer
            except Exception as e:
                return question_key, f"Error: {str(e)}"
        batch_size = 3
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), MAX_WORKERS)) as executor:
                future_to_question = {
                    executor.submit(worker_query, question): question["key"] 
                    for question in batch
                }
                for future in concurrent.futures.as_completed(future_to_question):
                    question_key, answer = future.result()
                    results[question_key] = answer
                    update_progress()
    except Exception as e:
        st.error(f"Error in parallel query processing: {str(e)}")
    progress_bar.empty()
    progress_text.empty()
    return results

# --- Scoring Rubric generation/validation use deepseek, English only ---
def generate_scoring_rubric(retriever):
    with st.spinner("Analyzing documents for insights..."):
        insights = parallel_query_documents(analysis_questions, retriever)
    scoring_system_prompt = """
    # Malaysian Household Subsidy Eligibility Scoring System

    ## Task
    Create a data-driven scoring system (0-100 points) to determine INDIVIDUAL HOUSEHOLD eligibility for government subsidies in Malaysia. Your system must evaluate HOUSEHOLDS (not regions or states) on various poverty risk factors.

    The system must:
    1. Assess individual households for subsidy eligibility
    2. Assign points across multiple household characteristics (total: 100 points)
    3. Produce a final score between 0-100 for each household
    4. Define clear eligibility thresholds (e.g., scores above 70 qualify for full subsidies)
    
    ## IMPORTANT: This scoring system is for INDIVIDUAL HOUSEHOLDS
    This is NOT a system to evaluate regions, states, or macroeconomic conditions. Each household should be able to be evaluated individually against these criteria.

    ## Analysis Steps
    1. Identify key household-level poverty determinants from Malaysian data
    2. Assign weights to variables (total: 100 points)
    3. Define scoring categories for each variable at the household level
    4. Create a mathematical eligibility formula that produces a 0-100 score
    5. Set threshold scores for different subsidy levels

    ## Key Household Variables to Consider
    - Household size (number of people)
    - Household monthly income relative to PLI (RM2,589)
    - Geographic location (urban/rural and state)
    - Education level of household head
    - Employment status and occupation of household head
    - Age of household head
    - Gender of household head
    - Ethnicity of household
    - Number of dependents
    - Housing conditions
    
    ## Example: For clarity, your output might contain scoring like:
    - Household Income: 30 points (0 points if above 200% of PLI, 15 points if between 100-200% of PLI, 30 points if below PLI)
    - Household Size: 15 points (3 points per dependent, up to 15 points)
    - Location: 10 points (0 points for urban areas in developed states, 10 points for rural areas in high-poverty states)
    
    ## Malaysian Context From Analysis
    
    POVERTY DETERMINANTS:
    {determinants}
    
    POVERTY LINE INCOME DATA:
    {pli_data}
    
    HOUSEHOLD SIZE FACTORS:
    {household_size}
    
    STATE-SPECIFIC POVERTY DATA:
    {state_data}
    
    URBAN VS RURAL FACTORS:
    {urban_rural}
    
    ETHNIC GROUP FACTORS:
    {ethnic_data}
    
    EDUCATION FACTORS:
    {education}
    
    EMPLOYMENT/OCCUPATION FACTORS:
    {employment}
    
    AGE FACTORS:
    {age_factor}
    
    GENDER FACTORS:
    {gender_factor}
    
    EQUIVALENCE SCALES:
    {equivalence_scales}
    
    EXISTING PROGRAMS:
    {existing_programs}

    ## Required Output Components
    1. **Variable Selection Table**
       - List of household-level variables with assigned weights (out of 100)
       - Statistical justification for each weight based on correlation with household poverty

    2. **Scoring Categories**
       - For each variable, categories with point values for individual households
       - Clear thresholds defining each category (e.g., income below RM2,000 = 30 points)

    3. **Mathematical Formula**
       - Precise formula to calculate household eligibility score (use LaTeX notation for complex formulas)
       - Example calculations for typical households
       - Express the formula both in words and mathematical notation

    4. **Eligibility Thresholds**
       - Score ranges for different subsidy levels (e.g., 70-100 points = full subsidy)
       - Explanation of each threshold

    5. **Implementation Notes**
       - Required household data for assessment
       - Handling of edge cases
       - Disclaimer that this is a data-driven suggestion requiring policy review

    Remember to focus on HOUSEHOLD-LEVEL assessment only, not macroeconomic indicators. The system must evaluate INDIVIDUAL HOUSEHOLDS, not regions or states. Each Malaysian household should be able to be scored individually using this system.
    """
    with st.spinner("Generating comprehensive household subsidy eligibility scoring system..."):
        try:
            llm = ChatOllama(
                model=ANALYSIS_MODEL,  # deepseek
                base_url=OLLAMA_BASE_URL,
                temperature=0.05,
                max_tokens=4000
            )
            formatted_prompt = scoring_system_prompt.format(**insights)
            scoring_system = llm.invoke(formatted_prompt)
            with open(os.path.join(LOG_DIR, "scoring_system.md"), "w") as log:
                log.write(scoring_system.content)
            with st.spinner("Validating scoring system logic..."):
                validation_result = validate_scoring_rubric(scoring_system.content)
                st.info("Scoring System Validation Results:")
                st.write(validation_result)
            return insights, scoring_system.content
        except Exception as e:
            st.error(f"Error generating scoring system: {str(e)}")
            return insights, f"Error generating scoring system: {str(e)}"

def validate_scoring_rubric(rubric):
    validation_prompt = """
    Please carefully check the logical consistency of the following scoring system, especially:
    1. Is the income scoring reasonable (lower income should get higher score)?
    2. Is household size scoring reasonable (larger households should get higher score)?
    3. Is education scoring reasonable (lower education should get higher score)?
    4. Is location scoring reasonable (rural/poor areas should get higher score)?
    5. Is occupation scoring reasonable (high-risk jobs should get higher score)?
    If you find any logical issues, explain and suggest corrections.
    Scoring System:
    {rubric}
    """
    try:
        llm = ChatOllama(
            model=ANALYSIS_MODEL,  # deepseek
            base_url=OLLAMA_BASE_URL,
            temperature=0.05
        )
        prompt = ChatPromptTemplate.from_template(validation_prompt)
        chain = prompt | llm | StrOutputParser()
        validation_result = chain.invoke({"rubric": rubric})
        return validation_result
    except Exception as e:
        return f"Validation error: {str(e)}"

# --- History fix: ensure save and load works, all English ---
def get_analysis_history():
    history = []
    for file in os.listdir(DATA_DIR):
        if file.startswith('analysis_results_') and file.endswith('.pkl'):
            try:
                result_id = file.replace('analysis_results_', '').replace('.pkl', '')
                with open(os.path.join(DATA_DIR, file), 'rb') as f:
                    data = pickle.load(f)
                    history.append({
                        'id': result_id,
                        'timestamp': data['timestamp'],
                        'file_count': len(data['file_hashes']),
                        'file_hashes': data['file_hashes'],
                        'file_names': data.get('file_names', []),
                        'insights': data['insights'],
                        'rubric': data['rubric'],
                        'all_tables': data['all_tables']
                    })
            except Exception as e:
                print(f"Error loading history: {e}")
                continue
    return sorted(history, key=lambda x: x['timestamp'], reverse=True)

def save_analysis_results(file_hashes, file_names, insights, rubric, all_tables):
    results = {
        'file_hashes': file_hashes,
        'file_names': file_names,
        'insights': insights,
        'rubric': rubric,
        'all_tables': all_tables,
        'timestamp': datetime.now().isoformat()
    }
    result_id = hashlib.md5(("_".join(sorted(file_hashes))).encode()).hexdigest()
    result_file = os.path.join(DATA_DIR, f"analysis_results_{result_id}.pkl")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)
    return result_id

# Analysis questions list
analysis_questions = [
    {
        "question": "What are the main determinants of poverty at the HOUSEHOLD or INDIVIDUAL level in Malaysia according to the documents? Provide specific statistical evidence about the correlation of each factor with poverty status where available.",
        "key": "determinants"
    },
    {
        "question": "What is the current Poverty Line Income (PLI) in Malaysia at the HOUSEHOLD level? Provide specific monetary values by state and location (urban/rural) if available.",
        "key": "pli_data"
    },
    {
        "question": "What is the statistical relationship between household size and poverty in Malaysia? Include specific data on how poverty rates vary by household size.",
        "key": "household_size"
    },
    {
        "question": "What are the precise poverty rates across different states in Malaysia? Provide percentages for each state.",
        "key": "state_data"
    },
    {
        "question": "How does urban versus rural location affect poverty rates for HOUSEHOLDS in Malaysia? Provide specific rate differences with percentages.",
        "key": "urban_rural"
    },
    {
        "question": "What are the differences in poverty rates among ethnic groups in Malaysia at the HOUSEHOLD level? Provide specific percentages for each group.",
        "key": "ethnic_data"
    },
    {
        "question": "How does education level of the HOUSEHOLD HEAD correlate with poverty status in Malaysia? Include specific data showing poverty rates by education level.",
        "key": "education"
    },
    {
        "question": "What is the relationship between employment status/occupation of the HOUSEHOLD HEAD and poverty in Malaysia? Include specific poverty rates by occupation and employment status.",
        "key": "employment"
    },
    {
        "question": "How does the age of the head of household relate to poverty risk in Malaysia? Include specific poverty rates by age group.",
        "key": "age_factor"
    },
    {
        "question": "What is the relationship between gender of household head and poverty in Malaysia? Provide specific statistics.",
        "key": "gender_factor"
    },
    {
        "question": "What income-to-needs ratios or equivalence scales are used in Malaysia to adjust for household size in poverty measurements?",
        "key": "equivalence_scales"
    },
    {
        "question": "What government subsidies or assistance programs currently exist in Malaysia for poor households, and what are their eligibility criteria?",
        "key": "existing_programs"
    }
]

def extract_json_from_response(text):
    # Extract first {...}, and try to fix common format issues
    match = re.search(r'({[\s\S]*})', text)
    if not match:
        return None
    json_str = match.group(1)
    # Remove trailing extra comma
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    return json_str

def deepseek_score_with_rubric(profile, rubric):
    """
    Use deepseek to score a profile based on the recommended scoring rubric.
    """
    prompt = f"""
You are a policy expert. Here is the recommended household subsidy scoring rubric:

{rubric}

Now, given the following applicant profile, please:
1. Calculate the eligibility score (0-100) according to the rubric.
2. State the eligibility status (APPROVED, PARTIALLY APPROVED, REJECTED).
3. Suggest a subsidy amount in RM.
4. Provide a detailed explanation, referencing the rubric logic for each factor.

Applicant Profile:
- Name: {profile['name']}
- Location: {profile['location']}
- Household Size: {profile['household_size']}
- Monthly Income: RM{profile['monthly_income']}
- Occupation: {profile['occupation']}
- Age: {profile['age']}
- Gender: {profile['gender']}
- Education Level: {profile['education_level']}
- Ethnicity: {profile['ethnicity']}
- Number of Dependents: {profile['dependents']}
- Housing Situation: {profile['housing_situation']}

Respond ONLY with a valid JSON object, do not include any markdown or extra text.
"""
    llm = ChatOllama(
        model=ANALYSIS_MODEL,  # deepseek
        base_url=OLLAMA_BASE_URL,
        temperature=0.05
    )
    response = llm.invoke(prompt)
    raw = response.content
    try:
        json_str = extract_json_from_response(raw)
        if not json_str:
            raise ValueError("No JSON object found in output.")
        result = json.loads(json_str)
        return result
    except Exception as e:
        # Show original content in UI for manual check
        return {
            "score": "N/A",
            "status": "ERROR",
            "subsidy_amount": "RM0",
            "explanation": f"Failed to parse deepseek output: {e}\nRaw (for manual check):\n{raw}"
        }

def main():
    st.title("Subsidy Eligibility Criteria Generator")
    st.write("Upload PDF documents about poverty in Malaysia to analyze and generate subsidy eligibility criteria recommendations")
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'rubric' not in st.session_state:
        st.session_state.rubric = None
    if 'all_tables' not in st.session_state:
        st.session_state.all_tables = None
    if 'current_analysis_id' not in st.session_state:
        st.session_state.current_analysis_id = None

    uploaded_files = st.file_uploader("Upload relevant PDF documents (research papers, official reports, etc.)", 
                                     type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != [f.name for f in uploaded_files]:
            start_time = time.time()
            
            with st.spinner("Processing documents in parallel..."):
                file_hashes, all_tables, all_text_contents = batch_process_files(uploaded_files)
            
            processing_time = time.time() - start_time
            st.success(f"✅ Processed {len(uploaded_files)} documents in {processing_time:.2f} seconds")
            
            with st.spinner("Setting up document analysis..."):
                st.session_state.retriever = get_or_create_vectordb(all_text_contents, file_hashes)
                
                if not st.session_state.retriever:
                    st.error("Failed to create document retriever. Please check the logs and try again.")
                    return
            
            st.session_state.insights, st.session_state.rubric = generate_scoring_rubric(st.session_state.retriever)
            st.session_state.all_tables = all_tables
            
            result_id = save_analysis_results(
                file_hashes,
                [f.name for f in uploaded_files],
                st.session_state.insights,
                st.session_state.rubric,
                all_tables
            )
            st.session_state.current_analysis_id = result_id
            st.session_state.last_uploaded_files = [f.name for f in uploaded_files]
            
            st.success(f"Analysis results saved with ID: {result_id}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Current Analysis",
        "History",
        "Recommended Scoring Rubric",
        "Key Insights",
        "Custom Query",
        "Subsidy Eligibility Comparison"
    ])

    with tab1:
        if st.session_state.retriever and st.session_state.insights and st.session_state.rubric:
            st.header("Current Analysis")
            st.write(f"Analysis ID: {st.session_state.current_analysis_id}")
            file_names = st.session_state.get('current_analysis_file_names', st.session_state.get('last_uploaded_files', []))
            st.write(f"Files processed: {len(file_names)}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files Processed", len(file_names))
            with col2:
                st.metric("Tables Extracted", len(st.session_state.all_tables))
            with st.expander("View Processed Files"):
                for fname in file_names:
                    st.write(f"- {fname}")
        else:
            st.info("No current analysis available. Please upload files to begin analysis.")

    with tab2:
        st.header("Analysis History")
        history = get_analysis_history()
        
        if history:
            for analysis in history:
                with st.expander(
                    f"Analysis from {datetime.fromisoformat(analysis['timestamp']).strftime('%Y-%m-%d %H:%M')} ({analysis['file_count']} files)"
                ):
                    st.write(f"**Analysis ID:** {analysis['id']}")
                    st.write(f"**Files Processed:** {analysis['file_count']}")
                    st.write(f"**Tables Extracted:** {len(analysis['all_tables'])}")
                    st.write("**Processed Files:**")
                    for fname in analysis.get('file_names', []):
                        st.write(f"- {fname}")
                    if st.button("Load This Analysis", key=f"load_{analysis['id']}"):
                        with st.spinner("Loading analysis..."):
                            st.session_state.retriever = get_or_create_vectordb([], analysis['file_hashes'])
                            if st.session_state.retriever:
                                st.session_state.insights = analysis['insights']
                                st.session_state.rubric = analysis['rubric']
                                st.session_state.all_tables = analysis['all_tables']
                                st.session_state.current_analysis_id = analysis['id']
                                st.session_state.current_analysis_file_names = analysis.get('file_names', [])
                                st.success("Analysis loaded successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to load analysis. Please try again.")
        else:
            st.info("No analysis history available.")

    if st.session_state.retriever and st.session_state.insights and st.session_state.rubric:
        with tab3:
            st.markdown(st.session_state.rubric)
            st.download_button(
                label="Download Scoring Rubric as Markdown",
                data=st.session_state.rubric,
                file_name="subsidy_eligibility_scoring_rubric.md",
                mime="text/markdown"
            )
        
        with tab4:
            for q in analysis_questions:
                with st.expander(q["question"]):
                    st.markdown(st.session_state.insights.get(q["key"], "No data available"))
        
        with tab5:
            st.subheader("Ask Custom Questions About the Documents")
            custom_question = st.text_input("Enter your question about poverty or subsidy criteria in Malaysia:")
            
            if custom_question and st.button("Get Answer"):
                with st.spinner("Researching answer..."):
                    answer, relevant_docs = query_document(custom_question, st.session_state.retriever)
                    
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    with st.expander("View Sources"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {i+1}**")
                            st.write(doc.page_content)
                            st.markdown("---")
    else:
        st.info("Please upload PDF documents or load a previous analysis to begin.")
    
    # Add subsidy eligibility comparison tab
    with tab6:
        comparison_tab()

if __name__ == "__main__":
    main()
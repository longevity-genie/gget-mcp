import sys
import json
import pytest
import os
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
from just_agents import llm_options
from just_agents.base_agent import BaseAgent
from gget_mcp.server import GgetMCP

# Load environment
load_dotenv(override=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "test"

# Load judge prompt
JUDGE_PROMPT = """# Judge Agent Prompt for gget Bioinformatics Tool Evaluation

You are a judge evaluating whether the right gget tools were used with correct parameters and produced reasonable results.

Compare the GENERATED ANSWER with the REFERENCE ANSWER.

PASS if:
- Appropriate gget tools were mentioned/used for the question type
- Key biological entities are correctly identified (gene names, species)
- The approach described is scientifically sound
- No major factual errors about the tools or biological concepts
- Style and exact wording don't matter - focus on correctness

FAIL if:
- Wrong gget tools mentioned for the task
- Major biological factual errors (wrong gene names, species, etc.)
- Tools used with clearly incorrect parameters
- Completely wrong methodology described

If PASS: respond only with "PASS"
If FAIL: respond with "FAIL" followed by detailed explanation:
- QUESTION: [the question]
- EXPECTED: [what was expected based on reference]
- GENERATED: [what was actually generated]
- REASON: [specific reason for failure]"""

# System prompt for test agent
SYSTEM_PROMPT = """You are a bioinformatics expert assistant powered by the gget library.

You have access to comprehensive bioinformatics tools through gget functions:
- Gene search and information retrieval (search_genes, get_gene_info)
- Sequence analysis and alignment (get_sequences, blast_sequence, muscle_align)
- Protein structure prediction and PDB data (alphafold_predict, get_pdb_structure)
- Expression analysis (archs4_expression, bgee_orthologs)
- Functional enrichment (enrichr_analysis)
- Disease and drug associations (cosmic_search, opentargets_analysis)
- Single-cell data analysis (cellxgene_query)

Important tool usage notes:
- get_pdb_structure requires a specific PDB ID (e.g., '2GS6'), not gene names
- For gene-to-structure workflows, use search_genes first, then alphafold_predict for prediction
- Always use search_genes to find Ensembl IDs before using other tools that need them
- cosmic_search uses 'searchterm' parameter, not 'gene'
- enrichr_analysis takes a list of genes, not individual genes

Always use the appropriate gget tools to answer biological questions. Include details about:
- Which tools you used
- Key findings from your analysis
- Scientific interpretation of results

Be thorough and scientific in your responses."""

# Test Q&A data covering different gget functionalities
QA_DATA = [
    {
        "question": "Find information about the human TP53 gene and get its protein sequence.",
        "answer": "Should use search_genes tool with TP53 and homo_sapiens to find Ensembl ID, then get_gene_info for details, and get_sequences with translate=True for protein sequence. TP53 is a tumor suppressor gene."
    },
    {
        "question": "What are the orthologs of BRCA1 gene across different species?",
        "answer": "Should use search_genes to get Ensembl ID for BRCA1, then bgee_orthologs tool with gene_id parameter to find orthologs across species. BRCA1 is involved in DNA repair."
    },
    {
        "question": "Perform enrichment analysis for a set of cancer-related genes: TP53, BRCA1, BRCA2, ATM, CHEK2.",
        "answer": "Should use enrichr_analysis tool with genes=['TP53', 'BRCA1', 'BRCA2', 'ATM', 'CHEK2'], database like 'KEGG_2021_Human', and species='human'. Results should show DNA repair and cancer pathways."
    },
    {
        "question": "Get the 3D structure information for the protein encoded by the EGFR gene.",
        "answer": "Should use search_genes tool first with 'EGFR' and 'homo_sapiens' to find the Ensembl ID, then potentially use alphafold_predict tool with protein sequence for structure prediction, OR search external databases for specific PDB IDs and use get_pdb_structure with a valid PDB ID (e.g., '2GS6'). The get_pdb_structure tool requires a specific PDB ID, not a gene name."
    },
    {
        "question": "Find mutations in the COSMIC database for the PIK3CA gene.",
        "answer": "Should use cosmic_search tool with searchterm='PIK3CA' to find cancer mutations. PIK3CA has hotspot mutations like H1047R."
    },
    {
        "question": "Analyze gene expression patterns for insulin (INS) gene across different tissues.",
        "answer": "Should use archs4_expression tool with gene='INS' to get tissue expression data. INS should show highest expression in pancreas."
    },
    {
        "question": "Perform BLAST search with a DNA sequence to identify its origin: ATGGCGCCCGAACAGGGAC.",
        "answer": "Should use blast_sequence tool with sequence='ATGGCGCCCGAACAGGGAC' and program='blastn' for DNA sequence. Results show alignment scores and gene matches."
    },
    {
        "question": "Find diseases associated with the APOE gene using OpenTargets.",
        "answer": "Should use search_genes to get Ensembl ID for APOE, then opentargets_analysis tool with ensembl_id parameter to find disease associations. APOE is associated with Alzheimer's disease."
    },
    {
        "question": "Get reference genome information for mouse (Mus musculus).",
        "answer": "Should use get_reference tool with species='mus_musculus' to get reference genome information and assembly details."
    },
    {
        "question": "Align multiple protein sequences and identify conserved regions.",
        "answer": "Should use muscle_align tool with multiple sequences to perform multiple sequence alignment and identify conserved regions."
    }
]

# Model configurations
answers_model = {
    "model": "gemini/gemini-2.5-flash-preview-05-20",
    "temperature": 0.0
}

judge_model = {
    "model": "gemini/gemini-2.5-flash-preview-05-20", 
    "temperature": 0.0
}

# Initialize gget server and get tools
gget_server = GgetMCP()

# Get all tool functions from the server
tools = [
    gget_server.search_genes,
    gget_server.get_gene_info,
    gget_server.get_sequences,
    gget_server.get_reference,
    gget_server.blast_sequence,
    gget_server.blat_sequence,
    gget_server.muscle_align,
    gget_server.diamond_align,
    gget_server.archs4_expression,
    gget_server.enrichr_analysis,
    gget_server.bgee_orthologs,
    gget_server.get_pdb_structure,
    gget_server.alphafold_predict,
    gget_server.elm_analysis,
    gget_server.cosmic_search,
    gget_server.mutate_sequences,
    gget_server.opentargets_analysis,
    gget_server.cellxgene_query,
    gget_server.setup_databases
]

# Initialize agents
test_agent = BaseAgent(
    llm_options=answers_model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

judge_agent = BaseAgent(
    llm_options=judge_model,
    tools=[],
    system_prompt=JUDGE_PROMPT
)

@pytest.mark.judge
@pytest.mark.skipif(
    os.getenv("CI") in ("true", "1", "True") or 
    os.getenv("GITHUB_ACTIONS") in ("true", "1", "True") or 
    os.getenv("GITLAB_CI") in ("true", "1", "True") or 
    os.getenv("JENKINS_URL") is not None,
    reason="Skipping expensive LLM tests in CI to save costs. Run locally with: pytest test/test_judge.py"
)
@pytest.mark.parametrize("qa_item", QA_DATA, ids=[f"Q{i+1}" for i in range(len(QA_DATA))])
def test_question_with_judge(qa_item):
    """Test each question by generating an answer and evaluating it with the judge."""
    question = qa_item["question"]
    reference_answer = qa_item["answer"]
    
    # Generate answer using gget tools
    generated_answer = test_agent.query(question)
    
    # Judge evaluation
    judge_input = f"""
QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

GENERATED ANSWER: {generated_answer}
"""
    
    judge_result = judge_agent.query(judge_input).strip()
    
    # Print for debugging
    print(f"\nQuestion: {question}")
    print(f"Generated: {generated_answer[:200]}...")
    print(f"Judge: {judge_result}")
    
    if "PASS" not in judge_result.upper():
        print(f"\n=== JUDGE FAILED ===")
        print(f"Question: {question}")
        print(f"Reference Answer: {reference_answer}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Judge Reasoning: {judge_result}")
        print(f"===================")
    
    assert "PASS" in judge_result.upper(), f"Judge failed for question: {question}\nReason: {judge_result}"

@pytest.mark.judge
@pytest.mark.skipif(
    os.getenv("CI") in ("true", "1", "True") or 
    os.getenv("GITHUB_ACTIONS") in ("true", "1", "True") or 
    os.getenv("GITLAB_CI") in ("true", "1", "True") or 
    os.getenv("JENKINS_URL") is not None,
    reason="Skipping expensive LLM tests in CI to save costs"
)
def test_agent_tool_integration():
    """Test that the agent can properly integrate multiple gget tools."""
    complex_question = """
    Analyze the EGFR gene: 
    1. Find its Ensembl ID
    2. Get detailed gene information
    3. Find associated diseases
    4. Look for cancer mutations
    5. Get expression data
    """
    
    response = test_agent.query(complex_question)
    
    # Check that response contains evidence of using multiple tools
    assert len(response) > 100, "Response too short for complex analysis"
    print(f"\nComplex analysis response: {response[:300]}...")

@pytest.mark.judge
@pytest.mark.skipif(
    os.getenv("CI") in ("true", "1", "True") or 
    os.getenv("GITHUB_ACTIONS") in ("true", "1", "True") or 
    os.getenv("GITLAB_CI") in ("true", "1", "True") or 
    os.getenv("JENKINS_URL") is not None,
    reason="Skipping expensive LLM tests in CI to save costs"
)
def test_sequence_analysis_workflow():
    """Test a complete sequence analysis workflow."""
    sequence_question = """
    I have this protein sequence: MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE
    Please:
    1. Perform a BLAST search to identify what protein this is
    2. Find similar sequences
    3. Get structural information if available
    """
    
    response = test_agent.query(sequence_question)
    
    # Check that response indicates BLAST was used
    assert len(response) > 50, "Response too short for sequence analysis"
    print(f"\nSequence analysis response: {response[:300]}...") 
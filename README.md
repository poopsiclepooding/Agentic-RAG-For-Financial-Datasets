# Agentic Retrieval-Augmented Generation (RAG) System

## Overview
The **Agentic Retrieval-Augmented Generation (RAG) System** offers a novel **dual-architecture approach** that bridges the gap between domain-specific and generalized retrieval methods. It provides both **adaptability** and **efficiency**, making it suitable for handling structured and unstructured datasets.

---

## Installation Instructions

1. Download and unzip the code files (83_h3_pathway_endterm.zip)
2. Unzip `raptor.zip` in your working directory
3. Ensure there is a `data` folder in the working directory containing required documents for retrieval
3. Install Dependencies
    - `pip install -r requirments.txt`
4. Run the code using streamlit
    - `streamlit run app.py`

---
## Key Features

### 1. **Dual-Architecture Design**
- **Domain-Specific Architecture**: 
  - Optimized for structured datasets with interdependent categories.
  - Uses parallel subagents to reduce redundancy and retrieval cycles.
- **General Architecture**:
  - Tree-based retrieval for unstructured datasets.
  - Employs topological sorting and depth limits to prevent infinite loops.

### 2. **Dynamic Agent-Based Retrieval**
- Dynamically orchestrates sub-agents to explore datasets efficiently.
- Tailors exploration for diverse query types, ensuring optimal coverage.

### 3. **Enhanced Data Fusion and Summarization**
- Combines retrieved information into concise, token-efficient summaries.
- Balances detail and clarity, avoiding verbosity or incompleteness.

### 4. **Tool-Enhanced Generation**
- Incorporates tools like calculators, web search APIs, and external knowledge sources.
- Enhances accuracy and context for complex queries.

### 5. **Advanced Strategies**
- Includes RAPTOR indexing, multi-query RAG fusion, query decomposition, and stepback techniques.
- Ensures precise and adaptable responses for complex queries.

### 6. **Balanced Flexibility and Efficiency**
- Scales seamlessly across structured and unstructured datasets.
- Maintains a fine balance between efficiency and versatility.

---

# 📄 RAG+KG 🤖

- An interactive web-based AI chatbot that allows users to upload documents (PDF) and ask questions about their content. 
- Automatically construct knowledge graphs from unstructured documents using LLM-generated ontologies and HuggingFace embeddings.
- Provide a unified retrieval server that combines three complementary search methods—vector similarity search, graph traversal for relationship-based queries, and logical filtering for metadata/attribute constraints—all orchestrated by autonomous AI agent that dynamically determine optimal retrieval strategies based on query complexity.
  
## 🎨 Images

<img width="600" height="400" alt="img" src="https://github.com/user-attachments/assets/ed3fd791-5e4a-474f-a951-5b94d59ba540" />


## 🧠 Agent Workflow

- The Document-to-Graph pipeline works by:
  - Converting the Document to chunks using PyMuPdf for data extraction and LangChain for splittling.
  - Then LLMGraphTransformer class from langchain is used to convert these chunks into graph document that have ontologies, entities, and relationships.
  - These are then stored in Neo4j graph database as knowledge graph.
  - The splits are also embedded into the Neo4j graph database using HuggingFace embeddings(all-MiniLM-L6-v2).
    
- The Agegntic Retrieval System works by:
   - Have 3 different tools: vector_search, graph_traversal, logical_filter.
   - The agent is created using LangChain with ConversationalBufferMemory.
   - The agent have access to multistep reasoning and iterative refinement by setting the agent_type to OPENAI_FUNCTIONS.

## 🚀 Features

- 📎 Upload and parse multiple PDF documents 
- 💬 Ask natural language questions based on the document
- 🧠 Powered by a LangChain-based intelligent agent
- ⚡ Parses text, tables, headings, formulas using PyMuPdf.
- 🤖 Uses Gemini LLM.
- 🔍 **Document Ingestion Pipeline** with:
  - ✅ LLM-powered automatic ontology generation (entities, relationships, hierarchies)
  - 🧠 Automated knowledge graph construction with entity resolution 
  - 🎯 The knowledege graph and semantic embeddings are stored in Neo4j. 
- 🌐 Simple and responsive web UI (Flask + Vanilla JS)


---

## 🛠️ Tech Stack

| Component                | Library/Tool                             |
|--------------------------|------------------------------------------|
| Agent Workflow           | LangChain   |
| Graph Database     | Neo4j + HuggingFace Embeddings    |
| Document Parsing              | PyMuPDF |
| Web App         | Flask + Vanilla JS           |
|LLM              | gemini-2.5-flash          |

---

## 📦 Setup Instructions

You need to install the Neo4j desktop and start create a instance. The default username, password will work with this code. But for any thing else you need to specify the url, username, password for the database in qna2.py at line 337.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nise-r/RAG-KG
cd RAG-KG

# Set API key in file directly
# qna2.py line 337
gemini_api = "YOUR_GEMINI_API_KEY"
```


### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
cd flask_qna_ai
```

### 3️⃣ Run the Application
```bash
python app.py
```

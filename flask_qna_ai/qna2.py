from rapidocr import RapidOCR
from PIL import Image
import io
import base64
import time
import os
import pymupdf

import PIL

import re
import tabula
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings



log_messages = []
class Parser:
    def __init__(self):
        pass
        
    def _extract_using_ocr(self,page):
        print("...OCR called...",end="")
        img = page.get_pixmap()
        img_bytes = img.tobytes()
        image = Image.open(io.BytesIO(image_bytes))

        if image.width > image.height:
            image = image.rotate(90,expand=True)

        image = image.resize((400,800))
        result = engine(image)
        text = "\n".join(txt for txt in result.txts)
        
        return text

    def _extract_text_excluding_tables(self,page):
        tables = page.find_tables(strategy="lines_strict")
        table_bboxes = [table.bbox for table in tables]

        def is_inside_any_table_bbox(bbox):
            for table_bbox in table_bboxes:
                # print(table_bbox)
                if pymupdf.Rect(table_bbox).intersects(pymupdf.Rect(bbox)):
                    return True
            return False

        # Get all text blocks
        blocks = page.get_text("blocks")  
        filtered_text = [
            block[4] for block in blocks
            if not is_inside_any_table_bbox(block[:4])
        ]

        return "\n".join(filtered_text)

    def _extract_table_content(self,page):
        tables = page.find_tables()
        tables_list = [table.to_markdown() for table in tables]

        text = "\n".join(text for text in tables_list)

        return text
    def _get_table_from_pg(self,pdf_path,pg):
        tables = tabula.read_pdf(pdf_path,pages=str(pg+1),multiple_tables=True)
        return tables
    
    def _extract_formulas_from_text(self,text):
        formulas = []

        # 1. LaTeX inline math: $...$
        inline_latex = re.findall(r'\$(.+?)\$', text)
        formulas.extend([f.strip() for f in inline_latex])

        # 2. LaTeX display math: \[...\]
        display_latex = re.findall(r'\\\[(.+?)\\\]', text, flags=re.DOTALL)
        formulas.extend([f.strip() for f in display_latex])

        # 3. LaTeX equation environments
        env_latex = re.findall(r'\\begin{equation\*?}(.+?)\\end{equation\*?}', text, flags=re.DOTALL)
        formulas.extend([f.strip() for f in env_latex])

        # 4. LaTeX align environments
        align_envs = re.findall(r'\\begin{align\*?}(.+?)\\end{align\*?}', text, flags=re.DOTALL)
        formulas.extend([f.strip() for f in align_envs])

        # 5. ASCII/Unicode math heuristics (e.g., x^2 + y^2 = z^2 or x² + y² = z²)
        # Look for lines with multiple math symbols or variables
        math_lines = []
        for line in text.splitlines():
            if re.search(r'[a-zA-Z0-9][\^²³√±*/=<>+\-]+[a-zA-Z0-9]', line):
                if len(line.strip()) > 5:  # avoid noise
                    math_lines.append(line.strip())

        # Filter duplicates and obvious non-formulas
        for line in math_lines:
            if line not in formulas and not line.startswith('Figure') and '=' in line:
                formulas.append(line)

        return formulas
    
    
    def _common_font_size(self,pdf_path):
        doc = pymupdf.open(pdf_path)
        font_sizes = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
        counter = Counter(font_sizes)
        return counter.most_common()[0][0]

    def _format_headings(self,headings):
        prev_y = 0
        result = ""
        for heading in headings:
            if heading['bbox'][1]!=prev_y:
                result += "\n"
            result+=heading['text']+" "
            prev_y = heading['bbox'][1]
        return result

    def _get_headings(self,page,comm_font_size):
        headings = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = round(span.get("size", 0))
                    font_flags = span.get("flags", 0)
                    text = span.get("text", "").strip()

                        # Skip empty strings
                    if not text:
                        continue

                        # Heuristic: large font size is probably a heading
                    if font_size > round(comm_font_size) or (font_size == round(comm_font_size) and (font_flags & pymupdf.TEXT_FONT_BOLD or "Bold" in span.get("font", ""))):
                        headings.append({
                            "text": text,
                            "size": font_size,
                            "font": span.get("font"),
                            "flags": font_flags,
                            "bbox": span.get("bbox"),
                        })

        return self._format_headings(headings)


    def parse_pdf(self,path):
        global log_messages
        log_messages.append("Parsing the pdf")
        doc = pymupdf.open(path)
        parsed = []
        comm_font_size = self._common_font_size(path)

        for i in range(doc.page_count):
            print(f"Page {i+1}",end="")

            full_pg = {}
            start_time = time.time()
            pg = doc.load_page(i)

            text = self._extract_text_excluding_tables(pg)
            img = ""
            table = ""
            
            if text == "" or text == []:
                text = self._extract_using_ocr(pg)
            else:
                table = self._get_table_from_pg(path,i)
                headings = self._get_headings(pg,comm_font_size)

            full_pg['text'] = text
            full_pg['tables'] = table
            full_pg['imgs'] = img
            full_pg['page'] = i+1
            full_pg['headings'] = headings
            full_pg['formulas'] = self._extract_formulas_from_text(text)
        
            parsed.append(full_pg)
            print(f"..Done.. {time.time()-start_time}")

        log_messages.append("PDF parsed")
        return parsed

class Agent:
    def __init__(self, gemini_api, url, username, password ,database):
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=gemini_api,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self._llm_graph = LLMGraphTransformer(
            llm=self._llm
        )
        self._graph =  Neo4jGraph(url=url, username=username, password=password,database=database)
        self._graph_chain = GraphCypherQAChain.from_llm(llm=self._llm,graph = self._graph,verbose = True, allow_dangerous_requests=True)
        self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        slef._db = Neo4jVector.from_documents(
            [], embedding=self._embeddings, url=url, username=username, password=password
        )
        self._vector_store = Neo4jVector.from_existing_index(
            self._embeddings,
            url=url,
            username=username,
            password=password,
            index_name="test",
            node_label="Chunk",          
            text_node_property="text",
            embedding_node_property="embedding",
            database=database
        )
        self._memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._parser = Parser()
        self._run = self._graph.query("""
                CREATE FULLTEXT INDEX entity IF NOT EXISTS
                FOR (e:__Entity__) ON EACH [e.id]
            """)
    
    def parse_and_save(self, document_path):
        global log_messages
        
        result = self._parser.parse_pdf(document_path)
        pdf = document_path
        docs_list = [Document(page_content=page['text']+'\n'.join(table.to_markdown() for table in page['tables'])
                          +"\nHeadings: "+ page['headings']+
                 '\n'.join(formula for formula in page['formulas']),

                metadata={"page": page['page'],"imgs":False if not page['imgs'] 
                  else ",".join(img.split('_')[1] for img in page['imgs']), 
                  'pdf_path':pdf,"headings":','.join(heading for heading in page['headings'].split('\n'))})
                 for page in result]
        
        doc_splits = self._text_splitter.split_documents(docs_list)
        graph_documents = self._llm_graph.convert_to_graph_documents(doc_splits)
        
        log_messages.append("Created the Knowledge Graph")
        
        self._graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True, 
            include_source=True
        )
        self._vector_store.add_documents(doc_splits)
        log_messages.append("Embedding Complete")
        
    def vector_search(self,query:str)->str:
        """
        Search for text similar to the query in vector database. Gives text chunk based on sentence similarity.
        """
        documents = self._vector_store.similarity_search(query,k=1)
        return "\n".join([doc.page_content for doc in documents])

    def graph_traversal(self,entity:str)->str:
        """
        Search for the relationship based on the entity in query. Gives the relationship stored in knowledge graph.
        This function is used to get relationship based data on entities.
        """
        response = self._graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
        return response
    
    

    def logical_filter(self,query):
        """
        Filter out the details based on some condition on the data stored in knowledge graph.
        """
        prompt = PromptTemplate.from_template("""
            Extract structured filters from this query.

            Query: {query}

            And generate the Cypher query to extract the data from Neo4j
            the keys in database are: {keys}

            The data is in text key.

            Strictly return the Cypher query only nothing else.
            """)
        keys = self._graph.query("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey")
        resp = self._llm.predict(prompt.format(query=query,keys=keys))
        if resp[:10] == "```cypher\n":
            resp =  self._graph.query(resp[10:-4])
        else:
            resp = self._graph.query(resp)
        return '\n'.join(text['n.text'] for text in resp)


    def create_agent(self):
        tools = [
            Tool(name="VectorSearch", func=self.vector_search, description="semantic match"),
            Tool(name="GraphTraversal", func=self.graph_traversal, description="relationship queries"),
            Tool(name="LogicalFilter", func=self.logical_filter, description="attribute filters"),
        ]
        
        agent = initialize_agent(tools,self._llm,agent_type=AgentType.OPENAI_FUNCTIONS,verbose=True,memory=self._memory)
        return agent,self._memory
    

gemini_api = "YOUR_API_KEY"

url="neo4j://127.0.0.1:7687"
username="neo4j"
password="password"


agent_inst = Agent(gemini_api,groq_api,url,username,password,database="neo4j")
agent, memory  = agent_inst.create_agent()

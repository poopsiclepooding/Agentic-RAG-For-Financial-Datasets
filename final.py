import os
import pandas as pd
import pathway as pw
import nltk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import json
import json_repair
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from swarm import Swarm,Agent
from langchain_openai import ChatOpenAI
import re
import json
import requests
from fuzzywuzzy import fuzz
from llm_guard import scan_prompt # pip install llm-guard
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity, Secrets, Language
from llm_guard.vault import Vault
import psutil
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from json import dumps, loads
import random
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from swarm import Swarm, Agent
from pathway.stdlib.indexing.data_index import DataIndex, InnerIndex
from pathway.stdlib.indexing.retrievers import AbstractRetrieverFactory,InnerIndexFactory
from pathway.stdlib.indexing.bm25 import TantivyBM25
from pathway.stdlib.indexing.nearest_neighbors import USearchKnn
from pathway.stdlib.indexing.hybrid_index import HybridIndex
from pathway.engine import BruteForceKnnMetricKind, USearchMetricKind
import logging
logging.basicConfig(level=logging.CRITICAL)

# # Run Once if ntlk error comes
# import nltk
# nltk.download('punkt', download_dir='/root/nltk_data')
# nltk.download('punkt_tab')

################## SET LICENSE AND KEYS ######################

pw.set_license_key('') # Pathway License Key
os.environ["GROQ_API_KEY"] = "" # Groq API Key
os.environ["OPENAI_API_KEY"] = ""


################## LLM GUARD INITIALIZATION ######################

vault = Vault()
input_scanners = [Anonymize(vault), Toxicity(), TokenLimit(), PromptInjection(), Secrets(), Language(valid_languages=["en"])]

os.environ["RUST_BACKTRACE"] = "full"


################## HYBRID SEARCH INDEX ######################

def default_usearch_hybrid_document_index(
    data_column: pw.ColumnReference,
    data_table: pw.Table,
    # perhaps InnerIndex should have a class that extends it with a promise it uses
    # data of fixed dimension
    dimensions: int,
    *,
    embedder: pw.UDF | None = None,
    metadata_column: pw.ColumnExpression | None = None,
) -> DataIndex:
    """
    Returns an instance of :py:class:`~pathway.stdlib.indexing.DataIndex`, with inner
    index (data structure) that is an instance of
    :py:class:`~pathway.stdlib.indexing.USearchKnn`. This method
    chooses some parameters of `USearchKnn` arbitrarily, but it's not necessarily a choice
    that works well in any scenario (each usecase may need slightly different
    configuration). As such, it is meant to be used for development, demonstrations,
    starting point of larger project, etc.

    Remark: the arbitrarily chosen configuration of the index may change (whenever tests
    suggest some better default values). To have fixed configuration, you can use
    :py:class:`~pathway.stdlib.indexing.DataIndex` with a parametrized instance of
    :py:class:`~pathway.stdlib.indexing.USearchKnn`.
    Look up :py:class:`~pathway.stdlib.indexing.DataIndex` constructor to see how
    to make data index parametrized by custom data structure, and the constructor
    of :py:class:`~pathway.stdlib.indexing.USearchKnn` to see the
    parameters that can be adjusted.

    """
    
    # BM25 Index for better "keyword" searching 
    bm25_index = TantivyBM25(   
        data_column=data_column,
        metadata_column=metadata_column,
    )

    # KNN Index for typical semantic embedding search
    knn_index = USearchKnn(
        data_column=data_column,
        metadata_column=metadata_column,
        dimensions=dimensions,
        reserved_space=1000,
        metric=USearchMetricKind.COS,
        embedder=embedder,
    )

    # Combined hybrid index
    inner_index = HybridIndex(
        retrievers=[knn_index, bm25_index],
        k=60
    )

    return DataIndex(
        data_table=data_table,
        inner_index=inner_index,
    )





################## GET DOCS, PARSER, SPLITTER ##################

# Monitor the folder /data for any new docs
folder = pw.io.fs.read(
    path="./data/",
    format="binary",
    with_metadata=True,
)

sources = [
    folder,
]


# PARSER #

from pathway.xpacks.llm import llms, parsers, prompts
parser = parsers.ParseUnstructured()
# # Uncomment this code if pdfs need to be handled
# chat = llms.OpenAIChat(model="gpt-4o")
# table_args = {
#    "parsing_algorithm": "llm",
#    "llm": chat,
#    "prompt": prompts.DEFAULT_MD_TABLE_PARSE_PROMPT,
# }
# image_args = {
#     "parsing_algorithm": "llm",
#     "llm": chat,
#     "prompt": prompts.DEFAULT_IMAGE_PARSE_PROMPT,
# }
# parser = parsers.OpenParse(
#     table_args=table_args, 
#     image_args=image_args,
#     processing_pipeline=None # defaults to CustomIngestionPipeline() defined in _openparse_utils, we can change this by defining our own pipeline from openparse
# )


# SPLITTER #

from pathway.xpacks.llm.splitters import TokenCountSplitter
splitter = TokenCountSplitter()
# splitter = None # If using OpenParse, then it handles the text splitting


# EMBEDDER #

from pathway.stdlib.indexing.data_index import DataIndex, InnerIndex
from pathway.stdlib.indexing.retrievers import AbstractRetrieverFactory,InnerIndexFactory
from pathway.stdlib.indexing.vector_document_index import default_usearch_knn_document_index
from pathway.xpacks.llm._utils import _coerce_sync
from pathway.xpacks.llm import embedders

embedder = embedders.SentenceTransformerEmbedder(
    model="intfloat/e5-large-v2",
    call_kwargs={}, # optional additional parameters to give to embedder
    device="cpu",
)





################## FINAL RETRIVER DEFINTION ##################

class CustomRetrieverFactor(InnerIndexFactory):
    def build_inner_index():
        return
    
    def build_index(
        self,
        data_column: pw.ColumnReference,
        data_table: pw.Table,
        metadata_column: pw.ColumnExpression | None = None,
    ) -> DataIndex:
        
        self.embedder = embedder
        self.embedding_dimension = len(_coerce_sync(self.embedder.__wrapped__)("."))
        hybrid_index = default_usearch_hybrid_document_index(
            data_column, 
            data_table, 
            dimensions=self.embedding_dimension,
            metadata_column=metadata_column,
            embedder=self.embedder,
        )
        return hybrid_index

retriever_factory = CustomRetrieverFactor()

from pathway.xpacks.llm.document_store import DocumentStore
document_store = DocumentStore(
    *sources,
    retriever_factory=retriever_factory, 
    parser=parser, 
    splitter=splitter,
)





# """ IO Test """
# !printf "id,owner,pet\\n1,Alice,dog\\n2,Bob,dog\\n3,Alice,cat\\n4,Bob,dog" > dataset.csv

# import pathway as pw
# class InputSchema(pw.Schema):
#   owner: str
#   pet: str
# t = pw.io.fs.read("dataset.csv", format="csv", schema=InputSchema)
# pw.debug.compute_and_print(t, include_id=False)







################## DOCUMENT STORE SERVER ##################

RUST_BACKTRACE=1
host = "127.0.0.1" # Define Server Address
port = 8674 # Define Server Port

from pathway.xpacks.llm.servers import DocumentStoreServer
server = DocumentStoreServer(host=host,port=port,document_store=document_store) # Initialize Document Store Server
server.run(threaded=True, with_cache=True, cache_backend=pw.persistence.Backend.filesystem("./Cache")) # Start Server






################## COMPANY NAME FROM - QUERY ##################

def extract_companies_from_question(question):
    pattern = r'between\s+(.+?)\s+and\s+(.+?)\?'
    
    match = re.search(pattern, question)
    
    if match:
        # Extract the two companies
        company_1 = match.group(1).strip()
        company_2 = match.group(2).strip()
        return [company_1, company_2]
    else:
        return None


def are_mostly_equal(str1, str2, threshold=70):
    similarity = fuzz.ratio(str1, str2)
    return similarity >= threshold

def find_contract_filename(companies, company_contract_df):
    if len(companies) != 2:
        return "Error: Two companies are needed."

    # Normalize company names for case-insensitive matching (strip spaces and lower case)
    company1, company2 = [company.strip().lower() for company in companies]

    # Loop through each contract to check if the companies are involved
    for _, row in company_contract_df.iterrows():
        # Clean and normalize the party list in the row
        cleaned_parties = eval(row['Cleaned_Parties'])  # Convert string list to actual list
        cleaned_parties = [party.strip().lower() for party in cleaned_parties]

        if any(are_mostly_equal(company1, party) for party in cleaned_parties) and any(are_mostly_equal(company2, party) for party in cleaned_parties):
            return row['Filename']  # Return the corresponding contract filename
        # # Check if both companies are in the list of parties
        # if company1 in cleaned_parties and company2 in cleaned_parties:
        #     return row['Filename']  # Return the corresponding contract filename
    print(f"No contract found between the companies {companies}")
    no_contract_return = f"No contract found between the companies {companies}"
    return no_contract_return

# Example CSV file path
csv_file_path = 'metadata_filename_parties.csv'
company_contract_df = pd.read_csv(csv_file_path)

query = "What is the Agreement Date for the contract between Birch First Global Investments Inc. and Mount Kowledge Holdings Inc.?"

companies = extract_companies_from_question(query)
if companies:
    print(f"The two companies are: {companies}")
    contract_filename = find_contract_filename(companies, company_contract_df)
    print(contract_filename.split(".pdf"))
    l = contract_filename.split(".pdf")
    l.append('.txt')
    contract_filename = ''.join(l)
    print(''.join(l))
    print(f"The contract filename is: {contract_filename}")
else:
    print("No companies found in the query.")



################## COMPANY NAME FROM - LLM INFERENCE ##################

company_template = """
You are smart assistant that helps find company names from sentences.
Given the sentence, give the companies involved in json format.
Example is provided below.
You must generate the output in a JSON containing a list with JSON objects.
Don't add any text at start or end, just give the json string.
Don't starting or ending ".

EXAMPLE:
Sentence : "What is the Minimum Commitment for the contract between HEALTHCARE CAPITAL CORP. and C.M. OLIVER & COMPANY LIMITED?"
Answer : {example_ans}

For the following sentence find the comapnies in it.
SENTENCE:
{sentence}
YOUR ANSWER:"""

example_ans = [
        {
            "companies":["HEALTHCARE CAPITAL CORP." , "C.M. OLIVER & COMPANY LIMITED"],
        },
    ]

company_prompt = ChatPromptTemplate.from_template(company_template) # Create promt template

company_model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

chain_company = company_prompt | company_model

def get_company_from_sentence(sentence: str):
    company_context = chain_company.invoke({"sentence" : sentence, "example_ans": example_ans})
    try:
        company_json = json_repair.loads(company_context.content)
    except:
        print("ERROR: Reponse of company finding model gives error")
    
    return company_json[0]["companies"]




    
############## DOCUMENTSTORE CLIENT ####################


class DocumentStoreClient:
    """
    A client you can use to query DocumentStoreServer.

    Please provide either the `url`, or `host` and `port`.

    Args:
        host: host on which `DocumentStoreServer </developers/api-docs/pathway-xpacks-llm/documentstore#pathway.xpacks.llm.document_store.DocumentStoreServer>`_ listens
        port: port on which `DocumentStoreServer </developers/api-docs/pathway-xpacks-llm/documentstore#pathway.xpacks.llm.document_store.DocumentStoreServer>`_ listens
        url: url at which `DocumentStoreServer </developers/api-docs/pathway-xpacks-llm/documentstore#pathway.xpacks.llm.document_store.DocumentStoreServer>`_ listens
        timeout: timeout for the post requests in seconds
    """  # noqa

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        timeout: int | None = 600,
        additional_headers: dict | None = None,
    ):
        err = "Either (`host` and `port`) or `url` must be provided, but not both."
        if url is not None:
            if host or port:
                raise ValueError(err)
            self.url = url
        else:
            if host is None:
                raise ValueError(err)
            port = port or 80
            self.url = f"http://{host}:{port}"
        self.timeout = timeout
        self.additional_headers = additional_headers or {}

    def query(
        self,
        query: str,
        k: int = 3,
        metadata_filter: str | None = None,
        filepath_globpattern: str | None = None,
    ) -> list[dict]:
        """
        Perform a query to the vector store and fetch results.

        Args:
            query:
            k: number of documents to be returned
            metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.
            filepath_globpattern: optional glob pattern specifying which documents
                will be searched for this query.
        """

        data = {"query": query, "k": k}
        # print("lol")
        # print(data)

        query = data['query']
        # print(query)
        companies = get_company_from_sentence(query)
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # companies = True
        # print(companies)

        if companies:
            # Find filepath
            filepath_globpattern = find_contract_filename(companies, company_contract_df)#, embedder)
            
            # If no contract found between the two companies
            if filepath_globpattern[:11]=="No contract": 
                return 0
            
            # Convert pdf to txt 
            if filepath_globpattern[-4:]==".pdf": 
                filepath_globpattern = filepath_globpattern.split(".pdf")[0]#.replace('-', '_').replace('.', '_').replace(' ','_')
            if filepath_globpattern[-4:]==".PDF":
                filepath_globpattern = filepath_globpattern.split(".PDF")[0]
            filepath_globpattern = "**/" + filepath_globpattern + ".txt"
            
            if metadata_filter is not None:
                data["metadata_filter"] = metadata_filter#f"""contains(path, '{metadata_filter}')"""
            if filepath_globpattern is not None:
                data["filepath_globpattern"] = filepath_globpattern
            url = self.url + "/v1/retrieve"
            # print(data)
            response = requests.post(
                url,
                data=json.dumps(data),
                headers=self._get_request_headers(),
                timeout=self.timeout,
            )

            responses = response.json()
            return sorted(responses, key=lambda x: x["dist"])
        
        else:
            print("Weren't able to trace company")

    # Make an alias
    __call__ = query

    def _get_request_headers(self):
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(self.additional_headers)
        return request_headers
    

# Create client instance
client = DocumentStoreClient(
    host=host,
    port=port,
)






############ ADVANCED RAG TECHNIQUES - MULTI QUERY ############

llm = ChatOpenAI(temperature=0)

# Multiquery to generate 6 queries 
template = """You are an AI language model assistant. You will be given a category and a query, and your task will be to create a new question on the given category,
that is similar to the query. Question: {Question} and Category: {Category}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_similar_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


# Different categories of information in CUAD Data
cat_csv = pd.read_csv('category.csv', delimiter='|')
cat = cat_csv["Category"]


# Load a conversational model
def gen_prompt(category,query):
    new_query = generate_similar_queries.invoke({"Question":query,"Category":category})
    return new_query

# Functions 
def get_category(query):
    for c in cat:
        if query.lower().find(c.lower()) >= 0 :
            return c
    
def get_group(category):
    # print(category,type(category))
    print(cat_csv[cat_csv["Category"]==category]["Group"])
    group = cat_csv[cat_csv["Category"]==category]["Group"].item()
    if group != '-':
        return cat_csv[cat_csv["Group"]==group]["Category"].to_list()
    return category


def create_subtask_agent(i, task_description):
    return Agent(
        name=f"Subtask_Agent_{i}",
        instructions=f"You are a sub task that will be performing a subtask of the original problem. Your task is : {task_description[0]}",
        functions = [retrieve_from_query]
    )




################ FUNCTIONS FOR LLM OS ###################

def list_files(directory=None):
    directory = directory or os.getcwd()
    try:
        files = os.listdir(directory)
        return f"Files in '{directory}': {', '.join(files)}"
    except Exception as e:
        return f"Error listing files: {str(e)}"

def search_files(search_query, directory=None):
    directory = directory or os.getcwd()
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if search_query.lower() in file.lower():
                    result.append(os.path.join(root, file))
        return f"Found files: {', '.join(result)}" if result else "No files found."
    except Exception as e:
        return f"Error searching files: {str(e)}"

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def create_file(filename, content=""):
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"File '{filename}' created successfully."
    except Exception as e:
        return f"Error creating file: {str(e)}"

def delete_file(filename):
    try:
        os.remove(filename)
        return f"File '{filename}' deleted successfully."
    except Exception as e:
        return f"Error deleting file: {str(e)}"

import psutil

def get_system_resources():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return (
            f"CPU Usage: {cpu_usage}%\n"
            f"Memory Usage: {memory.percent}% ({memory.used / 1e9:.2f} GB used / {memory.total / 1e9:.2f} GB total)\n"
            f"Disk Usage: {disk.percent}% ({disk.used / 1e9:.2f} GB used / {disk.total / 1e9:.2f} GB total)"
        )
    except Exception as e:
        return f"Error fetching system resources: {str(e)}"


def estimate_rag_runtime(num_documents, avg_document_size):
    try:
        base_time = 0.5  # Need to cvhange
        size_factor = avg_document_size / 100  
        system_load = psutil.cpu_percent(interval=1) / 100
        estimated_time = num_documents * base_time * (1 + size_factor + system_load)
        return f"Estimated RAG runtime: {estimated_time:.2f} seconds for {num_documents} documents."
    except Exception as e:
        return f"Error estimating RAG runtime: {str(e)}"


def summarize_file(file_path):
    file_path = os.path.join('data', file_path)
    if not os.path.exists(file_path):
        return "No file"
    
    try:
        with open(file_path, 'r') as file:
            text = file.read()
    except Exception as e:
        return f"Error reading the file: {str(e)}"
    
    words = text.split()[:125]
    input_text = ' '.join(words)
    prompt = PromptTemplate(input_variables=["text"], template="Summarize the following text: {text}")
    llm = OpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        summary = chain.run(input_text)
        return summary
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def test_functions():
    print("Testing file system management functions:")
    print(list_files()) 
    print(search_files("example")) 
    
    test_filename = "test_file.txt"
    print(create_file(test_filename, "This is a test file."))  
    print(read_file(test_filename))  
    print(delete_file(test_filename)) 
    
    print("\nTesting system resource management functions:")
    print(get_system_resources()) 
    print(estimate_rag_runtime(num_documents=10, avg_document_size=200))
    
    print("\nTesting summarizing text files in the current directory:")
    print(summarize_file('CybergyHoldingsInc_20140520_10_Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.txt')) 

test_functions()


################ LLM OS #####################

LLM_OS = Agent(
    name = 'llm_os',
    instructions = """
    You are an AI agent simulating an LLM OS for managing a legal database consisting of PDF, TXT, and XML files. You help system 
    administrators manage the database and also provide system resource information (CPU, memory, disk) and estimate RAG model runtime.
    Yopu should answer any query regarding the database, as well as perform a few functions on it like search, list, delete, etc. 
    You support the following tasks:
    List Files: List all files in a directory - list_files(directory)
    Search Files: Search for files by name. - search_files(search_query, directory)
    Read File: Read contents of a file - read_file(filename)
    Create File: Create a new file with content - create_file(filename, content)
    Delete File: Delete a specified file - delete_file(filename)
    Get Resources: Get current CPU, memory, and disk usage - get_system_resources()
    Estimate RAG Runtime: Estimate runtime for RAG model - estimate_rag_runtime(num_documents, avg_document_size)
    Summarize File: Summarize the first 125 words of a file (if it exists in the data folder) - summarize_file(file_path)
    """,
    functions = [list_files,search_files,create_file,read_file,delete_file,get_system_resources, estimate_rag_runtime, summarize_file],
)




def process_admin_query(query: str) -> str:
    """
    Processes a single query through the RAG pipeline and returns the assistant's response.

    Args:
        query (str): The user query.

    Returns:
        str: The assistant's response.
    """
    # Original query example

    # Apply LLMGuard to sanitize query input
    query, results_valid, results_score = scan_prompt(input_scanners, query)
    if any(not result for result in results_valid.values()):
        print(f"Prompt {query} is not valid, scores: {results_score}")
        return "Inappropiate input"
        exit(1)

    print(f"Guarded Prompt: {query}")
    try:
    

        c = Swarm()
        messages = [{"role":"user","content":f"{query}"}]
        response = c.run(agent=LLM_OS, messages = messages)

        return response.messages[-1]['content']
        # return response or "No answer generated from the summarized context."
    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"






def break_and_assign_task(list_category,query):
    chunks = []
    for i in range(len(list_category)):
        category = list_category[i]
        task_description = gen_prompt(category, query)
        agent = create_subtask_agent(i,task_description)
        response = c.run(
        agent=agent,
        messages=[{"role": "user", "content": f"{task_description}"}],
        )
        chunks.append(response.messages[-1]["content"])
    return chunks


main_chunk_retrievers = Agent(
    name = 'main_retriever',
    instructions = """ You are the main retriever agent as part of a RAG system. 
    The RAG system is having access to legal or financial files, and so to retrieve certain information 
    you might need other information too. The categories that are interdependent on each other 
    are called groups. We have a list of all the groups for the current dataset. According to the query, you 
    will be spawning new sub agents that are entasked with retrieving chunks related to that particular category 
    alone, but the catgeory wil belong to the group. You need to identify which of the categories the query 
    is asking for, and using a function find out all the categories belonging to its group. Then spawn sub 
    agents that retrieve each category.""",
    functions = [get_category, get_group, create_subtask_agent, break_and_assign_task],
)


def reciprocal_rank_fusion(results, k=60):
    """Reciprocal Rank Fusion for a flat list of text chunks and similarity scores."""
    fused_scores = {}

    for rank, (text_chunk, similarity_score) in enumerate(results):  # Iterate through results
        doc_str = dumps(text_chunk)  # Serialize text_chunk for use as a dictionary key
        if doc_str not in fused_scores:
            fused_scores[doc_str] = 0
        # Update fused score with the RRF formula and incorporate similarity_score
        fused_scores[doc_str] += (1 / (rank + k)) * abs(similarity_score)

    # Rerank results based on fused scores
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results



############ ADVANCED RAG TECHNIQUES - MULTI QUERY ############

llm = ChatOpenAI(temperature=0)

# Multiquery to generate 6 queries 
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


############ ADVANCED RAG TECHNIQUES - QUERY DECOMPOSITION ############

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

def process_question(question,  decomposition_prompt, llm, q_a_pairs):
    # Retrieve context
    context = retrieve_from_query(question,k=1)
    
    # Apply decomposition prompt
    decomposition_result = decomposition_prompt({
        "context": context[0],
        "question": question,
        "q_a_pairs": q_a_pairs
    })
    
    # Generate answer using LLM
    answer = llm(decomposition_result)
    
    # Format the QA pair
    q_a_pair = format_qa_pair(question, answer)
    
    # Append the new QA pair to the existing QA pairs
    return q_a_pairs + "\n---\n" + q_a_pair


#######################################################################################################################
## NEED TO ADD TO FULL RAG IMPLEMENTATION
# q_a_pairs = ""
# questions = generate_queries_decomposition.invoke({"question":question})
# for q in questions:
    
#     rag_chain = (
#     {"context": itemgetter("question") | retriever, 
#      "question": itemgetter("question"),
#      "q_a_pairs": itemgetter("q_a_pairs")} 
#     | decomposition_prompt
#     | llm
#     | StrOutputParser())

#     answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
#     q_a_pair = format_qa_pair(q,answer)
#     q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair





############ ADVANCED RAG TECHNIQUES - QUERY STEP BACK ############


examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

# Need to integrate:
# generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
# question = "What is task decomposition for LLM agents?"
# generate_queries_step_back.invoke({"question": question})




############ SUMMARIZING AGENT ############

summary_template = """
You are a summarizing Agent, which is tasked with summarizing a set of documents into 1 compiled text that  can be provided
to a generator as context, each with an associated ranking score. Documents with higher scores should contribute more to 
the summary, with less of their content being trimmed. Prioritize the higher-ranked documents, ensuring they have a 
stronger presence in the final summary. Write one combined paragraph including major parts of each document/chunk.
Here are the documents with their scores:
DOCUMENTS:
{context}
YOUR ORGANIZED DOCUMENTS:"""

summary_prompt = ChatPromptTemplate.from_template(summary_template) # Create promt template

# model = ChatOpenAI()
summary_model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

chain_summarize = summary_prompt | summary_model





############ GENERATOR ############

generator_template = """
You are smart assistant that helps users with their documents on Google Drive and Sharepoint.
Given a context, respond to the user question.
Give short sentence answers.
If there is no answer from context then say There is no mention of this clause
CONTEXT:
{context}
QUESTION: {question}
YOUR ANSWER:"""

generator_prompt = ChatPromptTemplate.from_template(generator_template)

# llm = ChatOpenAI()
generator_model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

chain_generator = generator_prompt | generator_model




############ FULL RAG PIPELINE ############

def RAG(question: str, k=1):

    # Multiquery to generate 4 queries  - Done
    queries = generate_queries.invoke({"question":question})
    random.shuffle(queries)
    multi_query_retrieve=[]
    for i in [2,3]:
        multi_query_retrieve.append(retrieve_from_query(queries[i],1))

    # Take 2 random apply query decomposition - Done
    decomposed_queries = generate_queries_decomposition.invoke({"question":question})
    decompose_retrieve = []
    for subq in decomposed_queries:
        decompose_retrieve.append(retrieve_from_query(subq,1))

    
    # Take 2 out of 4 and apply query step back - Done
    step_back_retrieve = [] # The retrieved chunks of step_back
    for i in [0,1]:
        step_back_query = generate_queries_step_back.invoke({"question": queries[i]})
        step_back_retrieve.append(retrieve_from_query(step_back_query,1))


    # Agentic Approach Retrieval - Done
    c = Swarm()
    messages = [{"role":"user","content":f"{question}"}]
    response = c.run(agent=main_chunk_retrievers, messages = messages)
    agentic_answer = [response.messages[-1]['content']]

    # Take all queries, recursively get all the retrieved documents
    retrieved_chunks = multi_query_retrieve + decompose_retrieve + step_back_retrieve #+ hyde_retrieve

    print("########  retrieved chunks: ######## ", retrieved_chunks)

    # Rerank the documents
    reranked_chunks = reciprocal_rank_fusion(retrieved_chunks[0])

    print("########  reranked chunks: ######## ", reranked_chunks)

    # Pass into the summarizer along with the retrieval from agentic part
    summarized_context = chain_summarize.invoke({"context" : reranked_chunks + agentic_answer})  

    print("########  summary chunks: ######## ", summarized_context.content)
    # pass the context into the generator
    response = chain_generator.invoke({"context" : summarized_context.content, "question" : question})
    
    return response





############# FUNCTIONS FOR MAIN AGENT ################

def retrieve_from_query(query: str, k: int = 3) -> list:

    retrieved = client.query(query, k)
    print(retrieved)
    retrived_context = []
    for retrieved_doc in retrieved:
        retrived_context.append((retrieved_doc["text"],retrieved_doc["dist"]))

    return retrived_context


def search_query(query):
    try:
        search_engine_url="https://www.google.com/"
        # Use search engine API to get results
        params = {
            "q": query,
            "key": " AIzaSyBJYmVmHNRk4aZT8Z9IXNsSZ4L09mIrl4M "
        }
        response = requests.get(search_engine_url, params=params)
        if response.status_code == 200:
            search_data = response.json()
            # Extract relevant information from search results
            results = [item['title'] + ": " + item['link'] for item in search_data.get('items', [])]
            return "\n".join(results) if results else "No relevant results found."
        else:
            return f"Search API request failed with status code {response.status_code}."
    except Exception as e:
        return f"An error occurred while using the search API: {str(e)}"


from datetime import datetime, timedelta

def arithmetic(operation, a, b):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b != 0:
            return a / b
        else:
            raise ValueError("Division by zero is not allowed.")
    else:
        raise ValueError("Unsupported arithmetic operation.")

def logical(operation, a, b=None):
    if operation == "AND":
        return a and b
    elif operation == "OR":
        return a or b
    elif operation == "NOT":
        return not a
    elif operation == "XOR":
        return (a or b) and not (a and b)
    else:
        raise ValueError("Unsupported logical operation.")

def compare_values(comparator, a, b):
    if comparator == "greater_than":
        return a > b
    elif comparator == "less_than":
        return a < b
    elif comparator == "equal":
        return a == b
    else:
        raise ValueError("Unsupported comparator.")

def date_difference(date1, date2, unit):
    d1 = datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.strptime(date2, "%Y-%m-%d")
    delta = d2 - d1
    if unit == "days":
        return delta.days
    elif unit == "weeks":
        return delta.days // 7
    elif unit == "months":
        return delta.days // 30
    else:
        raise ValueError("Unsupported unit for date difference.")

def add_to_date(date, value, unit):
    d = datetime.strptime(date, "%Y-%m-%d")
    if unit == "days":
        return (d + timedelta(days=value)).strftime("%Y-%m-%d")
    elif unit == "months":
        return (d + timedelta(days=value * 30)).strftime("%Y-%m-%d")
    elif unit == "years":
        return (d + timedelta(days=value * 365)).strftime("%Y-%m-%d")
    else:
        raise ValueError("Unsupported unit for adding to date.")

def subtract_from_date(date, value, unit):
    return add_to_date(date, -value, unit)

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


Arithmetic_Logic_Date_Agent = Agent(
    name='arithmetic_logic_date_agent',
    instructions="""
    You are an AI agent that provides basic arithmetic, logical operations, and date calculations. 
    You help users perform computations and analyze logic-based scenarios. 
    You support the following tasks:
    Arithmetic Operations: Perform basic arithmetic operations like addition, subtraction, multiplication, and division - arithmetic(operation, a, b)
    Logical Operations: Perform logical operations like AND, OR, NOT, XOR - logical(operation, a, b)
    Compare Values: Compare two values (greater than, less than, equal, etc.) - compare_values(comparator, a, b)
    Date Difference: Calculate the difference between two dates in days, weeks, or months - date_difference(date1, date2, unit)
    Add to Date: Add a specific number of days, months, or years to a date - add_to_date(date, value, unit)
    Subtract from Date: Subtract a specific number of days, months, or years from a date - subtract_from_date(date, value, unit)
    Current Date: Get the current date and time - get_current_date()
    """,
    functions=[arithmetic, logical, compare_values, date_difference, add_to_date, subtract_from_date, get_current_date],
)








################## FINAL PROCESSING QUERY FUNCTION ##################

def process_query(query: str) -> str:
    """
    Processes a single query through the RAG pipeline and returns the assistant's response.

    Args:
        query (str): The user query.

    Returns:
        str: The assistant's response.
    """
    # Original query example

    # Apply LLMGuard to sanitize query input
    query, results_valid, results_score = scan_prompt(input_scanners, query)
    if any(not result for result in results_valid.values()):
        print(f"Prompt {query} is not valid, scores: {results_score}")
        return "Inappropiate input"
        exit(1)

    print(f"Guarded Prompt: {query}")
    try:

        # Start the conversation with Swarm Agent based on the query
        print("swarm init")
        # arithematicsanddate = Swarm()
        # messages = [{"role": "user", "content": f"{query}"}]
        # response = arithematicsanddate.run(agent=Arithmetic_Logic_Date_Agent, messages=messages)
        response = RAG(question=query)
        print("got response")
        print(response)

        return response.content#.messages[-1]['content']
                
        # # Get retrieved text data
        # retrieved = client.query(query)
        # retrieved_context = []
        # retrieved_full = []
        # retrieved_full.append(retrieved)
        # for retrieved_doc in retrieved:
        #     retrieved_context.append(retrieved_doc["text"])
        
        # # Send the retrived data to generator model
        # arithematicsanddate = Swarm()
        # messages = [{"role": "user", "content": f"{query}"}]
        # response = arithematicsanddate.run(agent=Arithmetic_Logic_Date_Agent, messages=messages)

        # # Return the results
        # for index,message in response:
        #     for i in message :
        #         if i['content'] != None:
        #             return(i['content'])
        # print(response.messages[-1]['content'])
        # return response.messages[-1]['content'] or "Not Found."

    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"
























# """ Make the Retrieval A Function """




############ SUMMARIZING AGENT ############

# summary_template = """
# You are smart assistant that helps organize documents in proper english whilst keeping all information given to you.
# Given document text, organize it in proper sentences.
# DOCUMENTS:
# {context}
# YOUR ORGANIZED DOCUMENTS:"""

# summary_template = """
# You are a summarizing Agent, which is tasked with summarizing a set of documents into 1 compiled text that  can be provided
# to a generator as context, each with an associated ranking score. Documents with higher scores should contribute more to 
# the summary, with less of their content being trimmed. Prioritize the higher-ranked documents, ensuring they have a 
# stronger presence in the final summary. Write one combined paragraph including major parts of each document/chunk.
# Here are the documents with their scores:
# DOCUMENTS:
# {context}
# YOUR ORGANIZED DOCUMENTS:"""

# summary_prompt = ChatPromptTemplate.from_template(summary_template) # Create promt template

# # model = ChatOpenAI()
# summary_model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

# chain_summarize = summary_prompt | summary_model

# SINGLE_QUERY=1

# if SINGLE_QUERY:
#     summarized_context = chain_summarize.invoke({"context" : retrived_context})
#     print(f"This is the final context {summarized_context.content}")

# """ Generator """

# generator_template = """
# You are smart assistant that helps users with their documents on Google Drive and Sharepoint.
# Given a context, respond to the user question.
# Give short sentence answers.
# If there is no answer from context then say There is no mention of this clause
# CONTEXT:
# {context}
# QUESTION: {question}
# YOUR ANSWER:"""

# generator_prompt = ChatPromptTemplate.from_template(generator_template)

# # llm = ChatOpenAI()
# generator_model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

# chain_generator = generator_prompt | generator_model

# if SINGLE_QUERY:
#     response = chain_generator.invoke({"context" : summarized_context.content, "question" : queryy})
#     print(f"This is the final context: \n {response.content}")

# """ Accurary Of RAG Loop """

# test_data = pd.read_csv("contract_questions_and_answers.csv")
# data_dir_files = [f for f in os.listdir('data') if os.path.isfile(os.path.join('data', f))]

# semantic_embedder = SentenceTransformer('all-MiniLM-L6-v2')
# error_count = 0
# total_count = 0
# for index in test_data.index[0:2]:
#     if (test_data.Filename[index].split(".pdf")[0] + '.txt') not in data_dir_files and (test_data.Filename[index].split(".PDF")[0] + '.txt') not in data_dir_files:
#         # print(test_data.Filename[index].split(".pdf")[0] + '.txt')
#         continue
#     print(index)
#     # Get Question
#     question = test_data.Question[index]

#     # Perform retireval and summarization
#     retrived_context = retrieve_from_query(question, k=4)
#     summarized_context = chain_summarize.invoke({"context" : retrived_context})
    
#     # Perform generation
#     response = chain_generator.invoke({"context" : summarized_context.content, "question" : question})

#     # Perform sematic similarity with true answer
#     answer = (test_data.Answer[index].split("]")[0]).split("[")[1]
#     response_embedding = semantic_embedder.encode(response.content, convert_to_tensor=True)
#     answer_embedding = semantic_embedder.encode(answer, convert_to_tensor=True)
#     if answer != '':
#         similarity = cosine_similarity([response_embedding], [answer_embedding])[0][0]
#     else:
#         answer = "There is no mention of this clause"
#         answer_embedding = semantic_embedder.encode(answer, convert_to_tensor=True)
#         similarity = cosine_similarity([response_embedding], [answer_embedding])[0][0]

#     if similarity < 0.4:
#         error_count += 1
    
#     total_count += 1
#     print(f"Question : {question} \n Retireved Context : {retrived_context} \n Response : {response.content} \n Answer : {answer} \n Similarity : {similarity} \n ---------------------")
    
# correct_count = total_count - error_count
# accuracy = correct_count / total_count
# print(f"Accuracy is {accuracy}")
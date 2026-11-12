import os
import time
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Test dataset("Ground Truth")
EVALUATION_DATASET = [
    {
        "question": "What is the cold start time of QuantumCloud?",
        "expected_keyword": "under 5ms"
    },
    {
        "question": "Which consensus protocol is used for cross-region replication?",
        "expected_keyword": "ChronoSync"
    },
    {
        "question": "How much does outbound bandwidth cost?",
        "expected_keyword": "$0.15 per GB"
    },
    {
        "question": "What security model enforces access control across edge nodes?",
        "expected_keyword": "Zero-Trust"
    }
]

def run_benchmark(config_name: str, chunk_size: int, chunk_overlap: int, k_retrieved: int):
    print(f"\n==================================================")
    print(f" RUNNING {config_name}: chunk_size={chunk_size}, k={k_retrieved}")
    print(f"==================================================")

    # 1. Document load and split
    loader = TextLoader("sample_data.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    # 2.Connect container to ChromaDB via HTTP
    chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
    collection_name = f"eval_{config_name.lower().replace(' ', '_')}"
    
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=collection_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieved})

    # 3. CConfiguration of LLM and Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY using the provided context:\n\n{context}"),
        ("human", "{input}")
    ])
    chain = prompt | llm | StrOutputParser()

    # 4. Evauate questions
    hits = 0
    for item in EVALUATION_DATASET:
        q = item["question"]
        expected = item["expected_keyword"]

        start_time = time.time()
        retrieved_docs = retriever.invoke(q)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = chain.invoke({"context": context_text, "input": q})
        latency = time.time() - start_time

        is_hit = expected.lower() in context_text.lower()
        if is_hit:
            hits += 1

        status = "HIT" if is_hit else "MISS"
        print(f"[{status}] ({latency:.2f}s) Q: {q}")
        print(f"       Answer: {answer.strip()}")

        # Waiting to respect quota time 
        time.sleep(13)

    accuracy = (hits / len(EVALUATION_DATASET)) * 100
    print(f"\n--> {config_name} Retrieval Accuracy: {accuracy:.1f}% ({hits}/{len(EVALUATION_DATASET)})")
    return accuracy

if __name__ == "__main__":
    time.sleep(3)
    
    acc_a = run_benchmark("Config_A", chunk_size=120, chunk_overlap=10, k_retrieved=1)
    acc_b = run_benchmark("Config_B", chunk_size=400, chunk_overlap=50, k_retrieved=2)

    print("\n==================================================")
    print(" FINAL COMPARISON SUMMARY")
    print("==================================================")
    print(f"Config A (Small chunks, k=1): {acc_a:.1f}% Retrieval Accuracy")
    print(f"Config B (Larger chunks, k=2): {acc_b:.1f}% Retrieval Accuracy")
    print("==================================================")
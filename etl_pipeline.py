import os
import pdfplumber
import camelot
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# เปลี่ยนบรรทัดนี้จาก Chroma เป็น FAISS
from langchain_community.vectorstores import FAISS 

PDF_PATH = "แคตตาล็อค 593-2562 (ใหม่).pdf" # หาจากหน้าสุด
DB_DIR = "./faiss_db" # เปลี่ยนชื่อโฟลเดอร์เป็น faiss_db
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def clean_thai_ocr(text):
    if not isinstance(text, str): return ""
    return text.replace("SUU", "ระบบ").replace("1wwh", "มอเตอร์").replace("1ww", "มอเตอร์")\
               .replace("คอนโnsa", "คอนโทรล").replace("นน.", "น้ำหนัก").replace("กก.", "กิโลกรัม")

def extract_data_from_pdf(pdf_path):
    print(f"🔄 เริ่มดึงข้อมูลจากไฟล์: {pdf_path}")
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text: documents.append({"page": i+1, "content": clean_thai_ocr(text), "type": "text"})
    except Exception as e: print(f"❌ Error text: {e}")

    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for table in tables:
            df = table.df
            markdown_table = df.to_markdown(index=False)
            documents.append({"page": table.page, "content": clean_thai_ocr(markdown_table), "type": "table"})
    except Exception as e: print(f"⚠️ Error table: {e}")
    return documents

def build_vector_database(documents):
    print("🔄 กำลังสร้าง FAISS Vector Database...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chunks, metadatas = [], []
    
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        chunks.extend(splits)
        for _ in splits:
            metadatas.append({"source": "แคตตาล็อค 593-2562", "page": doc["page"], "data_type": doc["type"]})
            
    # ⚡️ โค้ดส่วนที่เปลี่ยนเป็น FAISS
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
    vectorstore.save_local(DB_DIR)
    print(f"🎉 สร้าง Database สำเร็จ! ลงในโฟลเดอร์ '{DB_DIR}'")

if __name__ == "__main__":
    extracted = extract_data_from_pdf(PDF_PATH)
    if extracted: build_vector_database(extracted)

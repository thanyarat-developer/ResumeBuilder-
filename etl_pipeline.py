import os
import pdfplumber
import camelot
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ================= การตั้งค่า (Configuration) =================
PDF_PATH = "data/แคตตาล็อค 593-2562 (ใหม่).pdf"
DB_DIR = "./chroma_db"
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
                if text:
                    documents.append({"page": i+1, "content": clean_thai_ocr(text), "type": "text"})
        print("✅ ดึงข้อความ (Text) สำเร็จ")
    except Exception as e: print(f"❌ Error: {e}")

    print("🔄 กำลังสแกนหาตารางในเอกสาร...")
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for table in tables:
            df = table.df
            markdown_table = df.to_markdown(index=False)
            documents.append({"page": table.page, "content": clean_thai_ocr(markdown_table), "type": "table"})
        print(f"✅ สกัดตารางสำเร็จ จำนวน {tables.n} ตาราง")
    except Exception as e: print(f"⚠️ ไม่พบตาราง: {e}")
    return documents

def build_vector_database(documents):
    print("🔄 กำลังสร้าง Vector Database...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chunks, metadatas = [], []
    
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        chunks.extend(splits)
        for _ in splits:
            metadatas.append({"source": "แคตตาล็อค 593-2562", "page": doc["page"], "data_type": doc["type"]})
            
    Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas, persist_directory=DB_DIR)
    print(f"🎉 สร้าง Database สำเร็จ! ลงในโฟลเดอร์ '{DB_DIR}'")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ ไม่พบไฟล์ {PDF_PATH} กรุณานำไฟล์ไปวางในโฟลเดอร์ data/")
    else:
        extracted_data = extract_data_from_pdf(PDF_PATH)
        if extracted_data: build_vector_database(extracted_data)

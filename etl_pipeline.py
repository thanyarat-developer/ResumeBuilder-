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
# ใช้โมเดล Embedding ที่รองรับภาษาไทยได้ดี
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def clean_thai_ocr(text):
    """
    ฟังก์ชันสำหรับ Data Cleansing แก้ไขปัญหาคำเพี้ยนจากการอ่าน OCR ภาษาไทย
    (โชว์ทักษะการจัดการกับ Data Quality)
    """
    if not isinstance(text, str): 
        return ""
    
    # แทนที่คำที่มักจะอ่านเพี้ยน
    cleaned = text.replace("SUU", "ระบบ") \
                  .replace("1wwh", "มอเตอร์") \
                  .replace("1ww", "มอเตอร์") \
                  .replace("คอนโnsa", "คอนโทรล") \
                  .replace("นน.", "น้ำหนัก") \
                  .replace("กก.", "กิโลกรัม")
    return cleaned

def extract_data_from_pdf(pdf_path):
    """ดึงข้อมูลทั้งแบบข้อความ (Text) และแบบตาราง (Table) ออกจาก PDF"""
    print(f"🔄 เริ่มดึงข้อมูลจากไฟล์: {pdf_path}")
    documents = []
    
    # 1. สกัดข้อความด้วย pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    cleaned_text = clean_thai_ocr(text)
                    documents.append({"page": i+1, "content": cleaned_text, "type": "text"})
        print("✅ ดึงข้อความ (Text) สำเร็จ")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อความ: {e}")

    # 2. สกัดตารางสเปกด้วย Camelot (ดึงทุกหน้าที่มีตาราง)
    print("🔄 กำลังสแกนหาตารางในเอกสาร...")
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for table in tables:
            df = table.df
            # แปลง DataFrame ของตารางเป็น Markdown เพื่อให้ AI เข้าใจโครงสร้าง Column/Row
            markdown_table = df.to_markdown(index=False)
            cleaned_table = clean_thai_ocr(markdown_table)
            documents.append({"page": table.page, "content": cleaned_table, "type": "table"})
        print(f"✅ สกัดตารางสำเร็จ จำนวน {tables.n} ตาราง")
    except Exception as e:
        print(f"⚠️ ไม่พบตาราง หรือเกิดข้อผิดพลาดในการดึงตาราง: {e}")
        
    return documents

def build_vector_database(documents):
    """หั่นข้อมูลและโหลดเข้าสู่ ChromaDB"""
    print("🔄 กำลังประมวลผลข้อมูล (Chunking) และสร้าง Vector Database...")
    
    # ตั้งค่าตัวหั่นข้อความ (Chunk Size 800 ตัวอักษร, Overlap กันประโยคขาด 100)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    chunks = []
    metadatas = []
    
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        chunks.extend(splits)
        
        # แนบ Metadata (สำคัญมากสำหรับการทำ RAG)
        for _ in splits:
            metadatas.append({
                "source": "แคตตาล็อค 593-2562 (ประตูม้วน)",
                "page": doc["page"],
                "data_type": doc["type"]
            })
    
    # โหลดลง Database และเซฟลงเครื่อง
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        metadatas=metadatas, 
        persist_directory=DB_DIR
    )
    # vectorstore.persist() # สำหรับ ChromaDB เวอร์ชันเก่า, เวอร์ชันใหม่ persist ให้อัตโนมัติ
    print(f"🎉 สร้าง Database สำเร็จ! โหลดข้อมูลทั้งหมด {len(chunks)} Chunks ลงใน '{DB_DIR}'")

if __name__ == "__main__":
    # ตรวจสอบว่ามีไฟล์ต้นทางหรือไม่
    if not os.path.exists(PDF_PATH):
        print(f"❌ ไม่พบไฟล์ {PDF_PATH} กรุณานำไฟล์ไปวางในโฟลเดอร์ data/")
    else:
        extracted_data = extract_data_from_pdf(PDF_PATH)
        if extracted_data:
            build_vector_database(extracted_data)
        else:
            print("❌ ไม่สามารถดึงข้อมูลได้ Pipeline หยุดทำงาน")

import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
# เปลี่ยนเป็น FAISS
from langchain_community.vectorstores import FAISS
import etl_pipeline 

st.set_page_config(page_title="Shutter Spec AI", page_icon="🏭", layout="centered")

DB_DIR = "./faiss_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PDF_PATH = "catalog.pdf"

# [ระบบสร้างฐานข้อมูลอัตโนมัติบน Cloud]
if not os.path.exists(DB_DIR):
    st.warning("ไม่พบฐานข้อมูล FAISS! ระบบกำลังสร้าง Knowledge Base อัตโนมัติ (อาจใช้เวลา 2-3 นาที)...")
    with st.spinner("AI กำลังอ่านและสกัดตารางสเปก..."):
        try:
            extracted_data = etl_pipeline.extract_data_from_pdf(PDF_PATH)
            if extracted_data:
                etl_pipeline.build_vector_database(extracted_data)
                st.success("สร้างฐานข้อมูลสำเร็จ! กำลังโหลดแอปพลิเคชัน...")
                st.rerun()
            else:
                st.error("❌ ไม่สามารถสกัดข้อมูลจาก PDF ได้")
                st.stop()
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการสร้างฐานข้อมูล: {e}")
            st.stop()

@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # โหลดไฟล์ FAISS พร้อมอนุญาตให้อ่านไฟล์ในเครื่องได้
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_database()

st.title("ระบบค้นหาสเปกประตูเหล็กม้วน (AI RAG)")
st.caption("เทคโนโลยี: FAISS Vector Search | พิมพ์ถามคำถามด้านล่างได้เลยค่ะ")

query = st.text_input("สอบถามข้อมูลสเปกสินค้า:", placeholder="เช่น ประตูทนไฟกันไฟได้นานกี่ชั่วโมง?")

if query:
    with st.spinner("AI กำลังค้นหา..."):
        results = vectorstore.similarity_search(query, k=3)
        if results:
            st.success("พบข้อมูลที่เกี่ยวข้องดังนี้:")
            for i, res in enumerate(results):
                with st.expander(f"📌 อ้างอิง {i+1} (หน้า {res.metadata['page']} - {res.metadata['data_type']})", expanded=(i==0)):
                    st.markdown(res.page_content)
        else:
            st.warning("ไม่พบข้อมูลค่ะ")

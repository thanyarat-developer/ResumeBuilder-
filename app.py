import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import etl_pipeline 

st.set_page_config(page_title="Shutter Spec AI", page_icon="🏭", layout="centered")

DB_DIR = "./faiss_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PDF_PATH = "catalog.pdf"

# [ระบบสร้างฐานข้อมูลอัตโนมัติบน Cloud]
if not os.path.exists(DB_DIR):
    st.warning("⚠️ ไม่พบฐานข้อมูล FAISS! ระบบกำลังสร้าง Knowledge Base อัตโนมัติ (อาจใช้เวลา 2-3 นาที)...")
    
    # ---------------- 🚨 โค้ดส่วนนักสืบ (Debug) 🚨 ----------------
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ ค้นหาไฟล์ชื่อ '{PDF_PATH}' ไม่เจอครับ!")
        st.info(f"📂 ไฟล์ทั้งหมดที่เซิร์ฟเวอร์มองเห็นตอนนี้คือ: {os.listdir('.')}")
        if os.path.exists("data"):
            st.info(f"📂 ไฟล์ในโฟลเดอร์ data/ มีดังนี้: {os.listdir('data')}")
        st.stop()
    # -----------------------------------------------------------
    
    with st.spinner("AI กำลังอ่านและสกัดตารางสเปก..."):
        try:
            extracted_data = etl_pipeline.extract_data_from_pdf(PDF_PATH)
            if extracted_data:
                etl_pipeline.build_vector_database(extracted_data)
                st.success("🎉 สร้างฐานข้อมูลสำเร็จ! กำลังโหลดแอปพลิเคชัน...")
                st.rerun()
            else:
                st.error("❌ หาไฟล์เจอครับ แต่มันดึงข้อความออกมาไม่ได้เลย (โปรดเช็คว่า PDF ล็อกรหัสผ่านไว้ หรือเป็นไฟล์รูปภาพล้วนๆ หรือไม่)")
                st.stop()
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดตอนอ่านไฟล์ (System Error): {e}")
            st.stop()

@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_database()

st.title("🏭 ระบบค้นหาสเปกประตูเหล็กม้วน (AI RAG)")
st.caption("เทคโนโลยี: FAISS Vector Search | พิมพ์ถามคำถามด้านล่างได้เลยครับ")

query = st.text_input("💬 สอบถามข้อมูลสเปกสินค้า:", placeholder="เช่น ประตูทนไฟกันไฟได้นานกี่ชั่วโมง?")

if query:
    with st.spinner("AI กำลังค้นหา..."):
        results = vectorstore.similarity_search(query, k=3)
        if results:
            st.success("พบข้อมูลที่เกี่ยวข้องดังนี้:")
            for i, res in enumerate(results):
                with st.expander(f"📌 อ้างอิง {i+1} (หน้า {res.metadata['page']} - {res.metadata['data_type']})", expanded=(i==0)):
                    st.markdown(res.page_content)
        else:
            st.warning("ไม่พบข้อมูลครับ")

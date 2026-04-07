import streamlit as st
import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import etl_pipeline 

st.set_page_config(page_title="Shutter Spec AI", page_icon="🏭", layout="centered")

DB_DIR = "./faiss_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ================= 🛡️ ส่วนการจัดการ API Key (แบบปลอดภัย) =================
st.sidebar.title("⚙️ ตั้งค่าระบบ AI")
# เปลี่ยนกลับมาใช้การรับค่าจากผู้ใช้แทนการฝังในโค้ด
gemini_api_key = st.sidebar.text_input("🔑 ใส่ Gemini API Key:", type="password")
st.sidebar.markdown("[👉 สมัครรับ API Key ฟรีที่นี่](https://aistudio.google.com/app/apikey)")
# =====================================================================

def find_pdf_file():
    pdf_files = glob.glob("*.pdf")
    return pdf_files[0] if pdf_files else None

PDF_PATH = find_pdf_file()

# [ระบบสร้างฐานข้อมูลอัตโนมัติ]
if not os.path.exists(DB_DIR):
    if PDF_PATH is None:
        st.error("❌ ไม่พบไฟล์ PDF ในระบบ!")
        st.stop()
    
    st.warning(f"⚠️ กำลังสร้างฐานข้อมูลจากไฟล์: {PDF_PATH}...")
    with st.spinner("AI กำลังอ่านและสกัดข้อมูล..."):
        try:
            extracted_data = etl_pipeline.extract_data_from_pdf(PDF_PATH)
            if extracted_data:
                etl_pipeline.build_vector_database(extracted_data)
                st.success("🎉 สร้างฐานข้อมูลสำเร็จ!")
                st.rerun()
            else:
                st.error("❌ สกัดข้อมูลไม่ได้")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_database()

st.title("ระบบค้นหาคู่มือฝ่ายขาย(AI RAG)")
if PDF_PATH:
    st.caption(f"กำลังใช้งานไฟล์: {PDF_PATH}")

query = st.text_input("💬 สอบถามข้อมูลสเปกสินค้า:", placeholder="เช่น ความหนา 0.7 ราคาเท่าไหร่?")

if query:
    if not gemini_api_key:
        st.warning("⚠️ กรุณาใส่ API Key ที่แถบด้านซ้ายเพื่อใช้งานระบบสรุปคำตอบโดย AI")
    else:
        with st.spinner("🔍 AI กำลังวิเคราะห์คำตอบ..."):
            results = vectorstore.similarity_search(query, k=3)
            if results:
                context_text = "\n\n".join([f"หน้าที่ {res.metadata['page']}:\n{res.page_content}" for res in results])
                
                prompt_template = """
                คุณคือผู้ช่วยเชี่ยวชาญด้านสเปกประตูเหล็กม้วน
                ข้อมูลอ้างอิง: {context}
                คำถาม: {question}
                คำตอบ:"""
                
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.1)
                    chain = prompt | llm
                    answer = chain.invoke({"context": context_text, "question": query})
                    
                    st.success("🤖 AI สรุปคำตอบ:")
                    st.markdown(f"**{answer.content}**")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("ไม่พบข้อมูลครับ")

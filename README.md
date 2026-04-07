# AI Sales Knowledge Assistant

**ระบบผู้ช่วยฝ่ายขายอัจฉริยะ**

โปรเจคนี้คือการพัฒนาระบบโดย ธันยรัศมิ์ ประภาจิรสกุล เป็นระบบ Data Pipeline แบบอัตโนมัติ (Automated RAG Pipeline) 
เพื่อดึงข้อมูลความรู้จาก "ไฟล์คู่มือ(PDF)" ซึ่งเป็น Unstructured Data นำมาจัดโครงสร้างใหม่และเก็บลงใน Vector Database 
เพื่อให้ AI สามารถค้นหารายละเอียดได้อย่างรวดเร็วและแม่นยำ

## ปัญหาทางธุรกิจที่แก้ไข (Business Problem & Impact)
* **ปัญหา:** ฝ่ายขาย ต้องการค้นหาข้อมูลต่างๆ ที่ใช้สำหรับตอบกลับลูกค้า ซึ่งคู่มือในไฟล์ PDF มีหลายหน้า ทำให้ล่าช้าและเสี่ยงต่อการตอบลูกค้าผิดพลาด
* **ทางแก้:** ทางผู้พัฒนาจึงพัฒนาระบบผู้ช่วยฝ่ายขายอัจฉริยะ AI Pipeline เพื่อดึงข้อความและ **ข้อมูลตารางต่างๆ** ออกจาก PDF อัตโนมัติ แปลงให้อยู่ในรูปแบบที่ AI ค้นหาได้ (Vector Embeddings) ลดเวลาค้นหาจากหลักนาทีเหลือเพียงไม่กี่วินาที

## สถาปัตยกรรมระบบ 

```mermaid
graph TD
    A[ข้อมูล: ไฟล์ PDF ต้องเป็นข้อความไม่ใช่รูปภาพ] --> B{Data Extraction}
    B -->|ถอดข้อความ| C[pdfplumber]
    B -->|ถอดตาราง| D[Camelot]
    
    C --> E[Data Cleansing]
    D -->|แปลงตารางเป็น Markdown| E
    
    E --> F[Text Splitter & Chunking]
    F -->|แนบ Metadata| G[HuggingFace Embeddings]
    
    G --> H[(Vector DB: ChromaDB)]
    H <--> I[Retrieval System]
    I <--> J[Web UI: Streamlit]

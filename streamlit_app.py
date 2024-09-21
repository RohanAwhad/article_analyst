import dataclasses
import openai
import os
import sys
import streamlit as st

from PIL import Image
import fitz
from pypdf import PdfReader
import pytesseract

@dataclasses.dataclass
class Message:
  role: str
  content: str | list[dict]

def llm_call(model: str, messages: list[Message]):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    res = client.chat.completions.create(model=model, messages=[dataclasses.asdict(x) for x in messages], temperature=0.8, max_tokens=4096)
    return res.choices[0].message.content

with open('./sys_prompt.txt', 'r') as f:
    sys_prompt = f.read().strip()

history = [Message('system', sys_prompt),]

def extract_text_using_tesseract(file_path):
    text = ''
    with fitz.open(file_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Identity)
            output = "output.png"
            pix.save(output)
            text += pytesseract.image_to_string(output)
    return text

def read_pdf(file_path):
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    text = ''
    for page in range(num_pages):
        text += reader.pages[page].extract_text()
    return text

def main():
    st.title("Article Analyst")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Generating summary..."):
            with open("uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            pdf_text = read_pdf("uploaded.pdf")

            if not pdf_text.strip():
                pdf_text = extract_text_using_tesseract("uploaded.pdf")

            if not pdf_text.strip():
                st.write("Sorry, couldn't extract text from the PDF.")
                return

            user_message = Message('user', pdf_text)
            history.append(user_message)
            response = llm_call(model="gpt-4o-mini", messages=history)
            st.markdown(response)

if __name__ == "__main__":
    main()

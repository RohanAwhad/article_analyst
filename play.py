import dataclasses
import openai
import os
import sys


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


import pytesseract
from PIL import Image
import fitz
import sys
from pypdf import PdfReader
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

pdf_text = read_pdf(sys.argv[1])
print(pdf_text)

if not pdf_text.strip():
    pdf_text = extract_text_using_tesseract(sys.argv[1])
    print(pdf_text)

if not pdf_text.strip():
    print('Sorry couldnt extract text from pdf')
    exit(0)

# create user message and call llm
user_message = Message('user', pdf_text)
history.append(user_message)
response = llm_call(model="gpt-4o-mini", messages=history)
print(response)

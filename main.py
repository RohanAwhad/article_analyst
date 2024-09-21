# Import necessary modules
import dataclasses  # For creating data classes
import openai  # For interacting with OpenAI's API
import os  # For interacting with the operating system
import sys  # For system-specific parameters and functions
from fastapi import FastAPI, File, UploadFile  # For creating a web API
from fastapi.responses import HTMLResponse  # For returning HTML responses
from pypdf import PdfReader  # For reading PDF files
import pytesseract  # For OCR (Optical Character Recognition)
from PIL import Image  # For image processing
import fitz  # For working with PDF files

# Initialize the FastAPI application
app = FastAPI()

@dataclasses.dataclass
class Message:
    role: str
    content: str | list[dict]

def llm_call(model: str, messages: list[Message]) -> str:
    """Call the OpenAI LLM with provided model and messages."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    res = client.chat.completions.create(
        model=model,
        messages=[dataclasses.asdict(x) for x in messages],
        temperature=0.8,
        max_tokens=4096
    )
    return res.choices[0].message.content

@app.get("/", response_class=HTMLResponse)
async def get_ui() -> str:
    """Serve the HTML interface for PDF text extraction and interaction with LLMs."""
    return '''
<!DOCTYPE html>
<html data-theme="forest">
<head>
    <title>PDF Text Extraction and LLM Interaction</title>
    <link href="https://cdn.jsdelivr.net/npm/daisyui@4.12.10/dist/full.min.css" rel="stylesheet" type="text/css" />
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue@2.6.14/dist/vue.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.26.1/axios.min.js"></script>
</head>
<body class="min-h-screen bg-base-200">
    <div id="app" class="container mx-auto py-8">
        <div class="hero">
            <div class="hero-content text-center">
                <div class="max-w-md">
                    <h1 class="text-5xl font-bold">PDF Text Extraction and LLM Interaction</h1>
                    <p class="py-6">Extract text from PDFs and interact with language models seamlessly.</p>
                    <input type="file" accept="application/pdf" @change="uploadPdf" class="block mb-4" />
                    <button @click="extractAndProcess" class="btn btn-primary mb-4">Extract and Process</button>
                </div>
            </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div v-if="extractedText" class="card bg-base-100 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title">Extracted Text</h2>
                    <p>{{ extractedText }}</p>
                </div>
            </div>
            <div v-if="llmResponse" class="card bg-base-100 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title">LLM Response</h2>
                    <p>{{ llmResponse }}</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        new Vue({
            el: '#app',
            data() {
                return {
                    extractedText: '',
                    llmResponse: ''
                }
            },
            methods: {
                async uploadPdf(event) {
                    const file = event.target.files[0]
                    const formData = new FormData()
                    formData.append('file', file)
                    
                    const response = await axios.post('/upload', formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        }
                    })
                    
                    this.extractedText = response.data.extractedText
                    this.llmResponse = response.data.llmResponse
                },
                async extractAndProcess() {
                    const response = await axios.post('/extract-and-process')
                    this.extractedText = response.data.extractedText
                    this.llmResponse = response.data.llmResponse
                }
            }
        })
    </script>
</body>
</html>
'''

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload a PDF file and save it locally."""
    contents = await file.read()
    with open("uploaded.pdf", "wb") as f:
        f.write(contents)
    return {"filename": file.filename}

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF."""
    try:
        reader = PdfReader(open(file_path, "rb"))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text using PyPDF2: {e}")
        return ""

def extract_text_using_ocr(file_path: str) -> str:
    """Extract text from a PDF file using OCR with pytesseract."""
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Identity)
                output = "temp.png"
                pix.save(output)
                text += pytesseract.image_to_string(Image.open(output))
        return text
    except Exception as e:
        print(f"Error extracting text using pytesseract: {e}")
        return ""

@app.post("/extract-and-process")
async def extract_and_process() -> dict[str, str]:
    """Extract text from the uploaded PDF and interact with the LLM."""
    text = extract_text_from_pdf("uploaded.pdf")
    
    if not text.strip():
        # Fallback to OCR extraction if PyPDF2 extraction fails
        text = extract_text_using_ocr("uploaded.pdf")
    
    if not text.strip():
        return {"error": "Failed to extract text from the PDF."}
    
    # Interact with the LLM using the extracted text
    with open('./sys_prompt.txt', 'r') as f:
        sys_prompt = f.read().strip()
    
    history = [Message('system', sys_prompt), Message('user', text)]
    response = llm_call(model="gpt-4o-mini", messages=history)
    
    return {"extractedText": text, "llmResponse": response}

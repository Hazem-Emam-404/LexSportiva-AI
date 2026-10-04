# LexSportiva-AI ⚽🏀🤾🎾🥊

**LexSportiva-AI** is a production-grade Multi-Sport Retrieval-Augmented Generation (RAG) assistant acting as an official AI sports referee. It answers complex rules and officiating queries grounded strictly in official international rulebooks with **exact claim citations and in-app PDF page viewing**.

---

## 🌟 Key Features

- **Official Rulebook Grounding**: IFAB (Football), FIBA (Basketball), IHF (Handball), ITF (Tennis), and IBA (Boxing).
- **Exact Page Citations**: Every rule claim is accompanied by the rulebook name and exact page reference (e.g., `[Football.pdf, p. 93]`).
- **In-App PDF Viewer**: Click any citation to open the original PDF at the exact cited page.
- **Hybrid Retrieval (Dense + Sparse)**: Combines `BAAI/bge-m3` dense embeddings and `BM25` keyword search with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranking.
- **Parent-Child Document Expansion**: Small child chunks are searched for precision, then expanded to full parent sections for coherent context.
- **Bilingual (English & Arabic)**: Native cross-lingual retrieval and natural response generation in both languages.
- **Persistent Chat History**: Gemini-style conversational interface backed by a SQLite database.
- **Cross-Domain Ready**: Client-side `X-User-ID` header and `localStorage` session handling for seamless deployment on Vercel and separate backend hosts.

---

## 🏛 Supported Sports & Rulebooks

| Sport | Governing Body | Official Rulebook |
|---|---|---|
| **Football (Soccer)** | IFAB | *Laws of the Game 2024/25* |
| **Basketball** | FIBA | *Official Basketball Rules 2024* |
| **Handball** | IHF | *Rules of the Game (Indoor Handball)* |
| **Tennis** | ITF | *ITF Rules of Tennis 2024* |
| **Boxing** | IBA | *Technical and Competition Rules* |

---

## ⚙️ Architecture & Pipeline

```text
  [ User Query (EN / AR) ]
             │
             ├──► Dense Search: BAAI/bge-m3 (ChromaDB)
             └──► Sparse Search: BM25
                          │
                          ▼
            [ Reciprocal Rank Fusion ]
                          │
                          ▼
         [ Cross-Encoder Reranker: mMiniLMv2 ]
                          │
                          ▼
        [ Parent Document Expansion (Docstore) ]
                          │
                          ▼
     [ Groq LLM: Llama-3.3-70b-versatile + Prompt ]
                          │
                          ▼
    [ Answer + Structured Citations + PDF Viewer (#page=X) ]
```

---

## 📁 Project Structure

```text
LexSportiva-AI/
├── backend/
│   ├── main.py                     # FastAPI REST API (Chat, SQLite history, PDF streaming)
│   ├── rag_pipeline.py             # SportsRAGPipeline (Groq LLM + Hybrid Search)
│   ├── database.py                 # SQLite database manager (conversations & messages)
│   ├── utils.py                    # Docstore loaders, BGE-M3 embeddings, prompt templates
│   ├── files/                      # Official rulebook PDFs (Football.pdf, etc.)
│   ├── sports_chroma_db_parent_child/ # Chroma vectorstore (child chunks)
│   ├── sports_parent_docs/         # Serialized parent document store
│   ├── datasets/                   # Golden evaluation datasets & questions
│   ├── notebooks/                  # Development & experimentation notebooks
│   ├── evaluate and train scripts/ # Ingestion and RAGAS evaluation scripts
│   ├── requirements.txt            # Python backend dependencies
│   └── .env                        # Backend environment variables
│
├── frontend/
│   ├── src/                        # React components (LandingPage, ChatView, CitationsSidebar, etc.)
│   ├── images/                     # AI Referee avatar assets
│   ├── index.html                  # HTML entry point with custom typography
│   ├── package.json                # Frontend dependencies & scripts
│   └── .env                        # Frontend environment variables (VITE_API_BASE_URL)
│
└── README.md
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+
- Groq API Key ([console.groq.com](https://console.groq.com))

---

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo GROQ_API_KEY=your_groq_api_key_here > .env

# Run FastAPI server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- API root: `http://127.0.0.1:8000`
- Swagger documentation: `http://127.0.0.1:8000/docs`

---

### 2. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env file
echo VITE_API_BASE_URL=http://localhost:8000/api > .env

# Start development server
npm run dev
```
- Access web interface at: `http://localhost:5173`

---

## 🌐 Deployment Guide (Vercel + Backend Host)

1. **Deploy Backend**: Deploy `backend/` to any cloud provider (e.g. Render, Railway, or VPS). Ensure CORS allows your frontend domain.
2. **Deploy Frontend on Vercel**:
   - Set the root directory to `frontend`.
   - Add the Environment Variable in Vercel project settings:
     ```env
     VITE_API_BASE_URL=https://your-backend-domain.com/api
     ```
3. **Session Persistence**: The frontend automatically generates and stores a persistent user identifier in `localStorage` and transmits it via the `X-User-ID` request header, preventing third-party cookie blocking across origins.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & pipeline status |
| `GET` | `/api/conversations` | List all conversation sessions for user (`X-User-ID`) |
| `POST` | `/api/conversations` | Create a new conversation session (`{"title": "..."}`) |
| `GET` | `/api/conversations/{id}` | Retrieve messages and citation list for a conversation |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation and its message history |
| `POST` | `/api/chat` | Send a prompt: `{"message": "...", "conversation_id": "..."}` |
| `GET` | `/api/pdf/{filename}` | Stream rulebook PDF with `#page={page}` direct anchor |

---

## 🛡 License
This project is open-source under the MIT License. Official rulebooks are property of their respective international federations (IFAB, FIBA, IHF, ITF, IBA).

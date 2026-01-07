# 🛡️ Imagery Guardrail Service

A production-grade AI microservice designed to validate prompts and images for safety, policy compliance, and domain relevance (specifically focused on Food AI). Built with **FastAPI**, **React**, and **Docker**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![React](https://img.shields.io/badge/react-18-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.113-green.svg)

---

## 🚀 Overview

The **Imagery Guardrail Service** acts as a security and quality layer for AI-driven image generation or processing pipelines. It ensures that inputs (text and images) adhere to safety guidelines and stay within the intended domain (Food/Culinary).

### Key Features
- **Text Guardrails**: Detects prompt injection, policy violations (NSFW, violence, hate), and domain relevance using `all-MiniLM-L6-v2`.
- **Image Guardrails**: Validates uploaded images for NSFW content and food relevance using `CLIP (ViT-B-32)`.
- **High Performance**: 
  - **Caching**: Redis-based result caching to avoid redundant ML inference.
  - **Lazy Loading**: Models are loaded on-demand to ensure fast service startup.
  - **Async Architecture**: Fully asynchronous processing using FastAPI.
- **Modern UI**: A sleek React-based dashboard for testing and monitoring guardrail decisions.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, SQLAlchemy (Async), Pydantic, Gunicorn/Uvicorn.
- **Frontend**: React (Vite), TailwindCSS, Shadcn/UI, React Query.
- **Infrastructure**: Docker, Redis, SQLite (Dev) / PostgreSQL (Prod).
- **AI/ML**: `Sentence-Transformers`, `Open-CLIP`, `PyTorch`.

---

## 📦 Installation & Setup

### Prerequisites
- Docker & Docker Compose
- Node.js 20+ (for local UI development)
- Python 3.11+ (for local backend development)

### 1. Quick Start with Docker (Recommended)
The easiest way to run the entire stack:

```bash
docker compose up -d
```

The service will be available at:
- **Backend API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Frontend UI**: `http://localhost:3000/ui/`

### 2. Local Development

#### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn src.main:app --reload --port 8000
```

#### Frontend
```bash
cd food-guard-ui
npm install
npm run dev
```

---

## 📖 API Documentation

### Validate Input
`POST /v1/guardrail/validate`

**Request Body:**
```json
{
  "prompt": "A delicious pepperoni pizza",
  "image_bytes": "base64_encoded_string_optional"
}
```

**Response:**
```json
{
  "status": "PASS",
  "reasons": [],
  "scores": {
    "injection_score": 0.0,
    "policy_score": 0.0,
    "food_domain_score": 0.98,
    "is_food_related": true
  },
  "metadata": {
    "processing_time_ms": 120,
    "cache_hit": false
  }
}
```

---

## 🏗️ Architecture Flow

1. **Request**: Client sends a prompt and/or image.
2. **Cache Check**: System computes a SHA-256 hash of the input and checks Redis.
3. **Validation Pipeline**:
   - **Text Check**: Scans for injection patterns and calculates domain similarity.
   - **Image Check**: (If image provided) CLIP model analyzes for NSFW and food relevance.
4. **Decision Engine**: Aggregates scores. If any threshold is breached, status is set to `BLOCK`.
5. **Logging**: Decision is logged to the database for audit trails.
6. **Response**: Returns the decision, reasons, and raw scores.

---

## 🧹 Maintenance

### Cleanup
To remove all containers and volumes:
```bash
docker compose down -v
```

---

## 📄 License
This project is licensed under the MIT License.

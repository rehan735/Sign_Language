# Quick Start Guide

## Prerequisites
- PostgreSQL installed and running
- Node.js 18+ installed
- Python 3.10+ installed (for the ISL prediction service)

## Setup Steps

### 1. Set Up Database
```bash
# Create database (run in PostgreSQL)
createdb accessible_comm_db

# Or using psql
psql -U postgres -c "CREATE DATABASE accessible_comm_db;"
```

### 2. Configure Database Connection
Edit `backend/.env` and update the DATABASE_URL with your PostgreSQL credentials:
```env
DATABASE_URL=postgresql://YOUR_USERNAME:YOUR_PASSWORD@localhost:5432/accessible_comm_db
```

### 3. Run Database Migrations
```bash
cd backend
npm run prisma:migrate
```

When prompted for migration name, enter: `init`

### 4. Start the Backend Server
```bash
npm run dev
```

Server will start at: `http://localhost:5000`

### 5. Start the Python Prediction Server
In a **second terminal** (keep the backend terminal running):
```bash
cd backend
venv\Scripts\python.exe predict_server.py
```

The prediction API will start at: `http://localhost:5001`

> **First launch takes ~30 seconds** while TensorFlow loads the Keras model.

Verify it's running:
```bash
curl http://localhost:5001/health
# → {"status":"ok","model_loaded":true}
```

### 6. Test Main Backend
Open a new terminal and run:
```bash
curl http://localhost:5000/health
```

You should see:
```json
{
  "success": true,
  "message": "Server is running",
  "timestamp": "..."
}
```

## Next: Test the API

Try registering a user:
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"Test User\",\"email\":\"test@example.com\",\"password\":\"password123\"}"
```

## View Database
```bash
npm run prisma:studio
```

This opens a GUI at `http://localhost:5555` to view your data.

---

For full API documentation, see `walkthrough.md`

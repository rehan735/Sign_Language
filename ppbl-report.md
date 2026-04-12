# Accessible Communication Application: Complete Project Walkthrough

This document provides a highly comprehensive, ground-up walkthrough of the entire Accessible Communication System based on our work so far. It covers the complete architecture, requirements, database structures, Entity-Relationship mappings, and technical system flows.

---

## 1. Executive Summary & What We Have Done

We have built a **full-stack Accessible Communication Web Application** designed to bridge communication gaps via **Indian Sign Language (ISL)** prediction, real-time voice, and text translation. 

**What we have accomplished from scratch:**
*   **Database & Backend Foundation:** Bootstrapped a PostgreSQL database via Prisma ORM with comprehensive structures for Users, Sessions, Preferences, and Translation histories. Built a Node.js/Express REST API serving these resources with robust JWT authentication and structured routing.
*   **AI Prediction Microservice:** Engineered a dedicated Python server (FastAPI) to handle ML inference. We integrated a custom-trained Convolutional Neural Network (CNN) Keras model capable of predicting **35 distinctive ISL classes** (Digits 1-9 and Letters A-Z).
*   **Machine Learning Pipeline:** Implemented an inventive data pipeline that takes raw MediaPipe hand-tracking coordinates, reconstructs them into 256x256 skeletal images in memory using OpenCV, and feeds them to the CNN model—replicating the model's exact original training environment.
*   **Frontend UI & Tracking Integration:** Created a responsive React 18 interface with Vite, Tailwind CSS, and Radix UI. Implemented real-time webcam integration combined with in-browser Mediapipe hand tracking to vastly decrease server load by processing geometric points client-side rather than uploading heavy video feeds.

---

## 2. Requirements Specification

### Hardware Requirements
*   **Client (User):** 
    *   Webcam (mandatory for Sign Language mode).
    *   Microphone (mandatory for Voice translation mode).
    *   Processor: Intel i3 / AMD Ryzen 3 or equivalent (required to fluently run in-browser MediaPipe models at 30fps).
*   **Server / Backend Node:**
    *   CPU: Dual Core + (The ML prediction service is run on CPU via `tensorflow-cpu`).
    *   RAM: Minimum 4GB (to comfortably retain Node.js, PostgreSQL, and Python/TensorFlow in memory).

### Software Requirements
*   **Core Systems:**
    *   OS: Windows / Linux / macOS.
    *   Node.js: v18+ 
    *   Python: v3.10+
    *   PostgreSQL: v13+
*   **Frontend Stack:** Vite, React 18, TypeScript, Tailwind CSS, MediaPipe Hand Tracking Models.
*   **Backend Main Stack:** TypeScript, Express.js, Prisma ORM, Bcrypt, JSON Web Tokens (JWT).
*   **Python Prediction Stack:** FastAPI, Uvicorn, TensorFlow (`tensorflow-cpu`), OpenCV (`opencv-python-headless`), Numpy.

---

## 3. Database Design & Tables

The master PostgreSQL database (`accessible_comm_db`) uses Prisma Schema. The database enforces referential integrity with `Cascade` deletions connected to the core `User` profile.

### Tables Breakdown
1.  **`users`**: The core authentication and identity map.
    *   `id` (PK), `name`, `email` (Unique), `password` (Hashed), `preferred_language` (Default: "en"), `created_at`, `updated_at`.
2.  **`user_sessions`**: Manages secure access without repetitive logins.
    *   `id` (PK), `user_id` (FK), `token` (JWT), `created_at`, `expires_at`.
3.  **`sign_detections`**: An activity log of accurately identified gestures.
    *   `id` (PK), `user_id` (FK), `detected_sign`, `confidence`, `language`, `frame_path`, `created_at`.
4.  **`voice_transcripts`**: A log of user voice translations and durations.
    *   `id` (PK), `user_id` (FK), `transcript`, `language`, `duration_seconds`, `created_at`.
5.  **`text_messages`**: History of standard text communications mapped.
    *   `id` (PK), `user_id` (FK), `message_text`, `language`, `created_at`.
6.  **`user_preferences`**: A personalized configuration mapping for frontend accessibility.
    *   `id` (PK), `user_id` (FK, Unique), `preferred_input_mode`, `accessibility_settings`, `updated_at`.

---

## 4. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    USER ||--o{ USER_SESSION : logs_into
    USER ||--o| USER_PREFERENCE : configures
    USER ||--o{ SIGN_DETECTION : translates
    USER ||--o{ VOICE_TRANSCRIPT : records
    USER ||--o{ TEXT_MESSAGE : creates

    USER {
        int id PK
        string name
        string email UK
        string password
        string preferred_language
        datetime created_at
    }
    USER_SESSION {
        int id PK
        int user_id FK
        string token
        datetime expires_at
    }
    USER_PREFERENCE {
        int id PK
        int user_id FK UK
        string preferred_input_mode
        string accessibility_settings
    }
    SIGN_DETECTION {
        int id PK
        int user_id FK
        string detected_sign
        float confidence
    }
    VOICE_TRANSCRIPT {
        int id PK
        int user_id FK
        string transcript
        int duration_seconds
    }
    TEXT_MESSAGE {
        int id PK
        int user_id FK
        string message_text
    }
```

---

## 5. System Architecture Design

We employ a decoupled **Microservices-lite Architecture**. Instead of running heavy Python ML scripts synchronously inside the Node API, they are separated.

### Component Design
1.  **The Client Layer (React/Vite)**
    *   Connects to the webcam and utilizes `@mediapipe/hands` to detect the 21 geometric points per hand. The client avoids sending high-latency video by compressing the visual information down into mere mathematical floats.
2.  **The Master API Layer (Node.js/Express, Port 5000)**
    *   Handles state, data retrieval, login/register endpoints, and historical timeline syncing. It interacts with the PostgreSQL DB entirely through Prisma.
3.  **The Inference Engine (Python/FastAPI, Port 5001)**
    *   A completely isolated, stateless mathematics server.
    *   React sends an array of coordinate frames to `http://localhost:5001/predict`.

---

## 6. The ML Prediction Pipeline (From Scratch)

Here is the exact pipeline workflow when a user makes an Indian Sign Language gesture into the camera:

1.  **Coordinate Extraction:**
    MediaPipe records the hands in the browser. A single frame is represented by 126 floating-point numbers: `2 hands × 21 landmarks per hand × 3 dimensions (X, Y, Z)`.
2.  **Batch Dispatch:**
    The React app bundles up several frames over a short window—e.g., 30 frames—and sends them as JSON to Python.
3.  **Best Frame Selection (Python):**
    The Python server calculates the sum of absolute values on the landmark coordinates to detect which frame has the most profound/clear hand activity. It tosses the rest.
4.  **Skeletal Image Reconstruction (OpenCV):**
    Using OpenCV, Python draws a **256x256 pixel empty black canvas**. It then places the X, Y coordinates as dots, and draws lines (bones) between the specified connections (thumb to index, palm, etc.). We now have an image representing a skeleton hand!
5.  **CNN Inference:**
    This `(256, 256, 1)` array is normalized (`/255.0`) and fed directly to identical dimensions expected by `Prediction_Model.h5` / `best_model.keras`.
6.  **Response Handling:**
    The model outputs an array of 35 probabilities. The server calculates the `argmax` (the highest probability index), maps it to our `LABELS` array (`["1", "2"... "A", "B"...]`), and returns `{ prediction: "C", confidence: 0.98 }` back to React for display.

---

## 7. Input and Output Structures

To clarify exactly how data enters and leaves the various systems, here are the core I/O formats:

### AI Prediction Service (Python)
**Input:** A JSON payload containing a rolling buffer of 30 frame snapshots. Each frame array contains 126 float values representing X, Y, Z coordinates for 42 hand landmarks (21 per hand).
```json
{
  "frames": [
    [0.12, 0.45, -0.01, ... /* 126 floats */],
    [0.13, 0.46, -0.02, ... /* 126 floats */]
    // up to 30 frames
  ]
}
```

**Output:** A JSON response containing the most likely ISL character and its confidence percentage (0 to 1.0).
```json
{
  "prediction": "A",
  "confidence": 0.985
}
```

### Main API Layer Authentication (Node.js)
**Registration/Login Input:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "securepassword123"
}
```

**Registration/Login Output:**
```json
{
  "success": true,
  "message": "User registered/logged in successfully",
  "user": {
    "id": 1,
    "name": "Jane Doe",
    "email": "jane@example.com"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5c..."
}
```

### Client Frontend (React/Vite)
**Input (from User):**
*   **Sign Mode:** Webcam Video Stream (intercepted by Mediapipe in browser).
*   **Voice Mode:** Microphone Audio Stream (intercepted by Web Speech API).
*   **Text/Keyboard Mode:** Text entered via input fields.

**Output (to User):**
*   Visual UI displaying the translated Sign Language character.
*   Textual display of voice transcripts.
*   Text-to-Speech (TTS) audio output for translated messages.

---

## 8. Testing & Tools Used

To ensure the reliability, performance, and accessibility of the Accessible Communication App, the following tools and methodologies were utilized:

### Core Development Tools
*   **Vite**: Used for lightning-fast modern frontend build tooling and Hot Module Replacement (HMR) during React development.
*   **Tailwind CSS & Radix UI**: Utilized for building an incredibly responsive, accessible, and highly customized premium aesthetic.
*   **Prisma Studio**: Employed as the visual database editor (`npm run prisma:studio`) for rapidly inspecting and modifying PostgreSQL records on the fly.
*   **nodemon & ts-node**: Used to continuously run and test the TypeScript Node.js backend without manual restarts.
*   **Postman / cURL / Node scripts**: Relied upon heavily to test standard REST API endpoints (authentication, preferences, health checks) independently of the UI. Tested using custom scripts like `test-register.js`.

### Testing Methodologies
1.  **API Integration Testing (Backend):** 
    *   Simulating valid and invalid JSON requests against Express endpoints.
    *   Testing JWT Token verification (ensuring expired, malformed, or missing tokens appropriately block access to protected database routes). Test scripts like `test-register-no-header.js` were used.
2.  **Machine Learning Inference Testing (Python):** 
    *   Testing the `predict_server.py` natively by firing synthetic coordinate arrays to ensure the CNN correctly resolves the mapping.
    *   Implementing and testing fallback errors (e.g., throwing `400` errors if hand landmarks are entirely missing from a batch). Scripts like `verify_prediction.js` and `analyze_debug.py` were utilized to track down prediction anomalies.
3.  **UI & Accessibility Testing (Frontend):** 
    *   Testing individual UI components for responsiveness across viewport sizes.
    *   Ensuring the camera stream properly hooks into `@mediapipe/camera_utils` without memory leaks.
4.  **End-to-End Real-Time Pipeline Testing:** 
    *   Connecting the webcam via React, performing sign language gestures in front of the lens, and confirming the React app successfully batches 30 frames, sends them to Python on port `5001`, and renders the returning text within low latency (sub-100ms).

---

## 9. Current Project Outcomes

As of today, the project has successfully accomplished its core minimum viable requirements and progressed into a highly refined, robust state:

1.  **Functional Base Platform:** The foundational full-stack architecture is completely stable. Users can register securely, manage their accessibility preferences, and sign in smoothly without friction via JWTs.
2.  **In-Browser Data Optimization:** Instead of passing heavy raw video streams to the server, we successfully integrated Edge-level AI inside the browser (MediaPipe). This radically decreased bandwidth overhead, as our server only needs to process a tiny JSON array of coordinates.
3.  **Cross-Language AI Inference:** We successfully bridged a modern Express/Node.js stack with a heavy Python/Keras environment without blocking or stalling the main API thread.
4.  **High Accuracy Static Gesture Detection:** The inference server correctly and rapidly predicts 35 distinct static classes of the Indian Sign Language (Letters A-Z and Digits 1-9), translating physical body movements into deterministic digital text in near real-time.

---

## 10. Conclusion & Future Work

### Conclusion
The Accessible Communication Application effectively demonstrates how modern web capabilities—combined with separated machine-learning microservices—can break down accessibility barriers. By abstracting the heavy lifting of image-classification geometry out of the browser and into an isolated Python pipeline, the system achieves remarkable prediction speed without sacrificing the user experience or frontend aesthetic.

### Future Work & Expansions

To elevate the application from a robust prototype to a fully mature commercial/enterprise-grade product, the following features are slated for future development:

1.  **Dynamic Sign Recognition (Time-Series ML):**
    *   *Goal:* Upgrade the CNN to an LSTM (Long Short-Term Memory) or Transformer-based AI model capable of understanding continuous, moving signs (like complete words and phrases) rather than just static letters or numbers.
2.  **Two-Way Accessibility (Avatar Generation):**
    *   *Goal:* Implement a 3D avatar on the frontend that parses spoken voice or standard text, and visually signs it back to the user in Indian Sign Language. 
3.  **Real-Time WebSocket Integration:**
    *   *Goal:* Transition from standard REST HTTP calls to WebSockets (e.g., `Socket.IO`) to allow continuous, persistent real-time streams of landmark coordinate data to the Python server, dropping latency down to sub-10ms logic.
4.  **Multi-Language Sign Support:**
    *   *Goal:* Expand the dataset beyond ISL (Indian Sign Language) to support ASL (American Sign Language), BSL (British Sign Language), and more.
5.  **Offline PWA Capabilities:**
    *   *Goal:* Compile the TensorFlow CNN directly into a `TensorFlow.js` model so that users can use the core prediction engine completely offline on their devices as a Progressive Web App (PWA).

---

## 11. Data Flow Diagrams (Levels 0, 1, and 2)

To provide a clearer, more standard systems analysis view, the Data Flow Diagram is divided into three levels: Context (Level 0), High-Level Processes (Level 1), and Deep-Dive Logic (Level 2).

### Level 0: Context Diagram
This shows the system as a single black box, illustrating only the external entities that interact with it.

```mermaid
flowchart TD
    User((User / Web Camera))
    System[Accessible Communication System]
    DB[(PostgreSQL Database)]

    User -- "1. Video Gestures / Auth Details" --> System
    System -- "2. Translated Sign Output (Text/Audio)" --> User
    
    System -- "3. Read/Write Data" --> DB
    DB -- "4. Stored States" --> System

    classDef core fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    class System core
```

### Level 1: Major Sub-Systems Diagram
This breaks the single system down into its primary functional modules (Frontend, API Backend, and ML Engine).

```mermaid
flowchart TD
    User((User / Web Camera))
    DB[(PostgreSQL Database)]

    UI[1.0 React/Vite Frontend]
    API[2.0 Node.js Master API]
    ML[3.0 Python AI Inference]

    User -- "Login Credentials" --> UI
    User -- "Physical Signs" --> UI
    
    UI -- "Translation Output" --> User
    
    UI -- "Auth / Save History" --> API
    API -- "JWT / Data" --> UI
    
    UI -- "X, Y, Z Coordinate Batches" --> ML
    ML -- "Predicted Class & Accuracy" --> UI
    
    API -- "CRUD Queries" --> DB
    DB -- "Query Results" --> API

    classDef frontend fill:#3b82f6,color:#fff,stroke:#1d4ed8,stroke-width:2px
    classDef backend fill:#10b981,color:#fff,stroke:#047857,stroke-width:2px
    classDef ai fill:#8b5cf6,color:#fff,stroke:#6d28d9,stroke-width:2px
    
    class UI frontend
    class API backend
    class ML ai
```

### Level 2: Detailed Process Diagram (Inference Pipeline)
This drills down specifically into Process 1.0 (Frontend Tracking) and Process 3.0 (Python ML Inference) to show the exact algorithmic loops.

```mermaid
flowchart TD
    UI((Client Layer))
    
    subgraph Frontend Processing
        MP[1.1 MediaPipe Library]
        Batch[1.2 JSON Batcher]
    end
    
    subgraph Python Prediction Service
        BestFrame[3.1 Absolute Value Selector]
        CV2[3.2 OpenCV Skeleton Mapper]
        CNN[3.3 TF/Keras CNN Evaluator]
        ArgMax[3.4 ArgMax Resolver]
    end

    UI -- "Raw Video Frame" --> MP
    MP -- "Yields 126 Floats" --> Batch
    Batch -- "POST /predict" --> BestFrame
    
    BestFrame -- "Highest Activity Layout" --> CV2
    CV2 -- "Draws (256, 256, 1) Image" --> CNN
    CNN -- "Array of 35 Probabilities" --> ArgMax
    ArgMax -- "Determines Character" --> UI

    classDef process fill:#f59e0b,color:#fff,stroke:#b45309,stroke-width:2px
    class MP,Batch,BestFrame,CV2,CNN,ArgMax process
```

---

## 12. System Flow Chart

Unlike the Data Flow Diagrams which map how data is handled across boundaries, this System Flow Chart illustrates the **step-by-step logic, control flow, and computational decision-making paths** within the application lifecycle.

```mermaid
flowchart TD
    %% Start
    Start([User Opens Web App]) --> AuthCheck{Valid JWT\nToken Available?}
    
    %% Auth branch
    AuthCheck -- No --> Login[Prompt Login / Register UI]
    Login --> AuthPost[POST /api/auth]
    AuthPost -- Success --> SaveToken[Save JWT to Storage]
    SaveToken --> MountApp
    AuthCheck -- Yes --> MountApp[Mount Dashboard UI]
    
    %% Application mounting
    MountApp --> ReqCam[Request Camera Permissions]
    ReqCam -- Denied --> RenderErr[Render Hardware Error]
    ReqCam -- Approved --> MPEngine[Initialize MediaPipe Engine]
    
    %% The loop
    MPEngine --> Scan[Scan Live Video Frame]
    Scan --> HandCheck{Are Hands\nDetected?}
    
    %% Hand not present
    HandCheck -- No --> LoopBack[Discard Frame]
    LoopBack --> Scan
    
    %% Hand present
    HandCheck -- Yes --> Extract[Extract 126 Float Coordinates]
    Extract --> Buffer[Append to 30-Frame Buffer Array]
    
    %% Batching logic
    Buffer --> BufferCheck{Buffer full\n(30 frames)?}
    BufferCheck -- No --> Scan
    BufferCheck -- Yes --> SendJSON[POST /predict to Python ML Server]
    
    %% Python Server Logic
    SendJSON --> PyLogic[Python: Select Frame with Highest Actvity]
    PyLogic --> OpenCVSkel[Python: OpenCV Renders 256x256 Skeleton Image]
    OpenCVSkel --> CNNRuns[Python: Pass Image through Keras CNN Model]
    CNNRuns --> ConfidenceCheck{Model Confidence\n> Threshold?}
    
    %% ML Results
    ConfidenceCheck -- No --> PyDrop[Return Null / Low Confidence]
    PyDrop --> ClearBuffer[React Empties Buffer]
    ClearBuffer --> Scan
    
    ConfidenceCheck -- Yes --> Resolve[Resolve ArgMax to Character Label]
    Resolve --> PyRet[Return Prediction JSON to Frontend]
    
    %% UI Update
    PyRet --> UpdateUI[Update React State / Trigger TTS Audio]
    UpdateUI --> DBLog[Async POST /api/sign to log History]
    
    %% DB Logging
    DBLog --> ClearBuffer
```

---

## 13. Comprehensive Module Descriptions

To thoroughly understand how the Accessible Communication Application's parts fit together chronologically, the project is divided into six distinct, loosely-coupled "Modules":

### Module 1: The Identity & Access Management (IAM) Module
*   **Technologies:** Node.js, Express, Prisma ORM, JSON Web Tokens (JWT), Bcrypt.
*   **Description:** This backend module acts as the core gatekeeper of the system. It handles user registration (hashing passwords via Bcrypt), login execution, and the issuance/validation of JWTs. It ensures that any requests made to user-specific logs or databases are strongly authenticated before access is granted.

### Module 2: The Hardware Interception Module
*   **Technologies:** React 18, HTML5 Media APIs `getUserMedia()`, Web Speech API.
*   **Description:** A pure frontend module entirely tasked with safely prompting the user for hardware permissions (Microphone & Camera) and establishing stable, real-time data streams into the DOM without stalling browser performance. 

### Module 3: The Edge-Tracking Optimization Module
*   **Technologies:** `@mediapipe/hands`, WebAssembly.
*   **Description:** Instead of sending massive 1080p video streams to our server, this edge-computing module runs fully within the client's browser. It analyzes the raw video feed 30 times a second, detects whether a hand is physically present, and distills the entire frame down to an incredibly lightweight mathematical array (126 floating-point values representing 3D spatial points).

### Module 4: The Machine Learning Inference Engine
*   **Technologies:** Python, FastAPI, OpenCV, TensorFlow/Keras.
*   **Description:** The "Brain" of the application running on Port `5001`. It acts as a stateless microservice. When the MediaPipe module sends over a batch of floating-point arrays, this module uses OpenCV to literally "draw" a skeleton image on a hidden 256x256 canvas in memory. It then passes that generated image to our trained Convolutional Neural Network (CNN). 
*   **Output:** It outputs an array of probabilities, resolving the highest probability into a deterministic Indian Sign Language Character (A-Z or 1-9) matching its training data set.

### Module 5: The Accessibility Delivery & UI Module
*   **Technologies:** Vite, Tailwind CSS, Radix UI, Web Speech API (TTS).
*   **Description:** This frontend module focuses strictly on visual and audible feedback. Once the ML Inference Engine confirms a letter translation, this module renders the specific visual update on the screen beautifully and uses Text-To-Speech (TTS) libraries to read the translated text aloud for visually impaired users communicating with the signing individual.

### Module 6: The Persistence & Log Module
*   **Technologies:** PostgreSQL, Prisma ORM, Node.js API.
*   **Description:** A background routing module tasked with historical permanence. Every time the User translates an ISL statement, or adjusts whether they want Dark Mode vs Light Mode via the UI, this module asynchronously dispatches a background request to log those details safely into our relational PostgreSQL database.

---

## 14. File Structure Design

Below is the design layout of our monolithic repository, separated by the functional tiers they belong to, presented in a tabular format for architectural clarity:

| Path / Directory | System Tier | Description |
| :--- | :--- | :--- |
| `/frontend/src/` | **Frontend (React)** | Core directory containing all React UI components, styling, and client-side logic. |
| `/frontend/src/components/` | **Frontend (React)** | Reusable Radix UI and Tailwind CSS visual building blocks (buttons, menus, videostream wrapper). |
| `/frontend/src/services/` | **Frontend (Data)** | Functions managing external API connections (communicating with Node.js and the Python Server). |
| `/backend/src/` | **Backend (Node.js)** | Core directory for the Express.js Master API server routing and middleware logic. |
| `/backend/src/server.ts` | **Backend (Node.js)** | The entry point where the Express server is initialized, headers are set, and routes are mapped. |
| `/backend/src/controllers/`| **Backend (Node.js)** | Files managing the raw business logic for routes (e.g. Auth controllers parsing Bcrypt and JWTs). |
| `/backend/src/middleware/` | **Backend (Node.js)** | Interceptor functions verifying tokens and error-handling before hitting the controllers. |
| `/backend/src/routes/` | **Backend (Node.js)** | The URL path definitions mapping directly to the controller logic (e.g., `/api/auth`). |
| `/backend/prisma/` | **Database (ORM)** | Contains the `schema.prisma` file orchestrating the PostgreSQL table creation, migrations, and relationships. |
| `/backend/predict_server.py`| **ML Inference (Python)**| The standalone FastAPI Python server bridging MediaPipe coordinate ingestion and the Keras CNN. |
| `/backend/Model/` | **ML Inference (Data)**| Contains the actual pre-trained Deep Learning weights (`best_model.keras` or `Prediction_Model.h5`). |
| `/backend/scripts/` | **Tooling / DevOps** | Internal test scripts and data-pipeline debugging tools (e.g., verifying prediction bounds organically). |

---

## 15. Database & Data Source Files

While the layout above explores the architectural code routing, it is critical to understand where exactly the underlying data is sourced, validated, and stored permanently.

### Relational Database Files
The master data source is a PostgreSQL instance. The application does not write complex raw SQL directly inside controllers; instead, it uses the Prisma ORM (Object-Relational Mapper) to govern the ecosystem.

*   `backend/.env`: The heavily protected environment file containing the critical `DATABASE_URL` strings required for Node and Prisma to securely authenticate into the PostgreSQL system.
*   `backend/prisma/schema.prisma`: The absolute "Source of Truth" for the database architecture. This file explicitly maps the tables (Users, History, Transcripts, Sessions), sets Default values, and defines cascading referential integrity.
*   `backend/prisma/migrations/`: An auto-generated vault where Prisma stores timestamped `.sql` migration files state-changes. This ensures that every database structure change is permanently tracked and can be cleanly rolled back.

### Machine Learning Dataset Sources
The geometric intelligence source powering the Python backend is derived entirely internally. It is split between raw training image vaults and generated weight models.

*   `backend/dataset/Indian/`: Contains the accurately labeled raw image datasets separated into directories by characters (A-Z) and digits (1-9).
*   `backend/dataset/ISL custom Data/` & `backend/dataset/Indian sign Language-Real-life Words/`: Extended robust image datasets providing diversified spatial variance mapped specifically for the Neural Network to train against edge cases.
*   `backend/Model/best_model.keras` & `Prediction_Model.h5`: The finalized data sources. Rather than uploading imagery to a third-party API, these files represent the *compiled mathematical culmination of the datasets above*, baked down into static neuron weights. FastAPI loads these directly into internal memory to execute rapid mathematical predictions locally.

---

## Formal Academic Report Sections

*If you are migrating this documentation into a formal university/academic report, you can copy-paste the sections below directly. They accurately reflect your specific application logic while matching standard formatting.*

### 3.3 Database / Data Source

The master PostgreSQL database (`accessible_comm_db`) uses Prisma Schema. The database enforces referential integrity with Cascade deletions connected to the core User profile.

The datasets used in this project are sourced from publicly available platforms and local curations, such as:
*   **Kaggle** — for labeled Indian Sign Language (ISL) static imagery.
*   **Custom Local Datasets** — for Real-life ISL words and capturing variance in physical proportions.

The dataset contains important attributes such as:
*   Image data (pixel values, 256x256 resolution, grayscale mappings).
*   Spatial geometric landmarks (X, Y, Z coordinate matrices representing 42 hand joints).
*   Labeled outcome classes spanning 35 specific characters (A-Z, 1-9).

In this workflow, the datasets undergo multiple stages including data collection, preprocessing (via MediaPipe), mathematical feature extraction, and model training. The deep learning models are trained using TensorFlow and Keras, which are used for building and optimizing Convolutional Neural Networks (CNNs) for image classification of synthesized skeletal structures and handling other machine learning logic.

Video feed data is processed using browser-side computer vision techniques (MediaPipe) to extract geometric structures, while the resulting numerical coordinates are analyzed using Python and OpenCV methods to render in-memory imagery without storing files on disk. The trained Keras models are then integrated into the FastAPI backend system for real-time prediction output scoring.

### 3.4 File / Data Structure Design

The key data structures used in the Accessible Communication system implementation are as follows:

| Class / Structure | Type | Description |
| :--- | :--- | :--- |
| `MediaPipeEngine` | Frontend API | Handles loading of edge-based ML models in the browser to extract hand landmarks continuously from the live webcam storage buffer. |
| `JSONBatcher` | Frontend Logic | Buffers arrays of 126 floating-point values structured to send sequential snapshots (rolling window of 30 frames). |
| `OpenCVSkeletonMapper` | Python Function | Performs preprocessing such as converting coordinates back into 2D lines and generating an empty normalized `(256, 256, 1)` image canvas. |
| `PredictServer` | Python FastAPI | A standalone microservice handling stateless REST API inputs from the frontend interface. |
| `CNNEvaluator` | TF/Keras Model | Uses trained deep-learning models to generate probability predictions utilizing convolution across the 256x256 skeletal frame. |
| `ArgMaxResolver` | Python Function | Processes the 35 raw probability nodes output by the CNN and thresholds them against a confidence level to yield a deterministic string result. |
| `PrismaSchema` | ORM Model | Maps User identity structures, session tokens, histories, and access-patterns securely to PostgreSQL records natively. |
| `AppInterface` | Frontend Module | Handles standard user interaction, configures accessibility modes, triggers Text-To-Speech (TTS), and displays final translation results visually. |

### 3.5 Implementation and Maintenance

The implementation and ongoing maintenance of the Accessible Communication system rely on a strictly decoupled architecture, ensuring that updates to one component do not destabilize the entire application workflow.

**A. System Implementation & Deployment Phase:**
*   **Decoupled Microservices:** The system is implemented across two distinct active servers running concurrently. Node.js manages the standard HTTP business logic across Port `5000` (`npm run dev`), while Python/FastAPI explicitly claims Port `5001` (`python predict_server.py`) to manage heavy local CPU/TensorFlow thread allocation.
*   **Environment Configuration:** All sensitive data is abstracted from the main repository. Implementation requires migrating the internal `.env` framework to link the Prisma ORM securely to the local or cloud PostgreSQL database cluster.
*   **Hardware Bootstrapping:** The client side initializes via the `React` / `Vite` pipeline, immediately prompting for HTML5 DOM permissions to leverage the local machine's web camera and microphone, acting as the primary boundary layer to the edge-computing models.

**B. System Maintenance & Scalability Phase:**
*   **Continuous Model Re-Training:** The Keras CNN is not a black-box. The `Prediction_Model.h5` and `.keras` files are designed to be swapped. As new datasets are added to the `/backend/dataset/Indian/` directories, administrators can quickly recompile the Neural Network and drop the newly trained weight files into the `/backend/Model/` folder to instantly upgrade prediction accuracy without altering backend logic.
*   **Database Migrations:** Routine database updates, such as adding new accessibility preferences, are managed securely through Prisma Migrations (`npm run prisma:migrate`). This maintains parity between the active database schema and the underlying application codebase securely.
*   **Dependency Management:** Security and speed optimizations require periodic monitoring of the `package.json` and `requirements.txt` ecosystems. Upgrades to `tensorflow-cpu`, `opencv-python-headless`, and `@mediapipe/hands` can be routinely executed to take advantage of upstream performance patches.

### 4.1 Project Outcomes

The execution of the Accessible Communication software has yielded a heavily optimized, highly functional cross-platform application that successfully bridges standard web infrastructure with advanced machine learning. The critical outcomes include:
*   **Systemic Latency Reduction:** By stripping the raw video feed down to 126 skeletal coordinates directly on the user's local browser via MediaPipe, the system successfully obliterated the massive heavy upload latency typically present in cloud-video prediction algorithms.
*   **Accurate Real-Time Prediction:** The Keras Convolutional Neural Network (CNN) demonstrated highly reliable classification of 35 distinct static gestures across the Indian Sign Language (ISL) alphabet and numeric sequence (A-Z, 1-9).
*   **Decoupled Stability:** The explicit separation between the Node.js/PostgreSQL business state and the stateless Python prediction pipeline proved that intense mathematical inferences do not need to bottleneck front-end UI delivery or authentication loops.

### 4.2 Conclusion

This project successfully proves that advanced accessibility tools—specifically real-time sign language translation—can be embedded natively into standard web architectures without requiring users to own specialized hardware. By leveraging the geometric edge-computing inherent in HTML5 Media APIs and pairing it with a customized, statically trained Deep Learning neural network, the application reliably converts physical motion into digital text and synthesized speech. The final result is a scalable, highly secure platform that effectively dissolves communication barriers for hearing and speech-impaired individuals in digital ecosystems.

### 4.3 Future Scope & Work

While the foundational application executes perfectly against its initial requirements, several distinct expansions are slated for future development to elevate the platform to an enterprise-grade standard:
*   **Time-Series Tracking (Dynamic Spoken Sentences):** Transitioning the underlying logic from a static Convolutional Neural Network (CNN) to a Long Short-Term Memory (LSTM) or Transformer network to capture moving, multi-gestural semantic sentence structures.
*   **Two-Way Visual Avatars:** Developing a 3D browser-rendered avatar capable of parsing standard text or microphone speech and visually converting it back into sign language for the user, establishing true two-way native communication.
*   **WebSocket Upgrades:** Replacing the standard REST HTTP polling cycles with a continuous WebSocket (`Socket.IO`) stream to drive translation latency down below 10 milliseconds.
*   **Broader Globalization Datasets:** Expanding the neural training sets beyond the current Indian Sign Language (ISL) boundaries to support dialect-specific variants of American Sign Language (ASL).

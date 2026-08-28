# 📚 Semantic Book Recommender

An NLP-based semantic book recommendation system that recommends books based on the meaning and context of a user's query rather than relying only on keyword matching.

The project uses Hugging Face sentence-transformer embeddings to represent book descriptions as vectors, Chroma for semantic similarity search, and Gradio to provide an interactive recommendation dashboard.

---

## 🚀 Features

- 🔍 Semantic search using natural-language queries
- 📚 Recommends books based on their descriptions
- 🧠 Hugging Face sentence-transformer embeddings
- ⚡ Chroma vector database for similarity search
- 🏷️ Fiction / Non-Fiction category filtering
- 😊 Emotional-tone based filtering
- 🖼️ Displays book covers
- ✍️ Displays book titles, authors, and descriptions
- 💻 Interactive Gradio dashboard

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Hugging Face
- Sentence Transformers
- Transformers
- LangChain
- Chroma
- Gradio
- Jupyter Notebook

---

## 🧠 How It Works

The recommendation system follows a semantic search pipeline.

### 1. Data Preparation

The book dataset contains information such as:

- ISBN
- Title
- Authors
- Categories
- Description
- Ratings
- Thumbnail

Missing book categories are predicted using a zero-shot classification model.

The categories are then simplified into broader groups such as:

- Fiction
- Non Fiction
- Children's Fiction
- Children's Non Fiction

---

### 2. Text Preparation

Book descriptions are processed and stored in:

```text
tagged_description.txt
```
---

### 3. Generate Embeddings

Each book description is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```
---

### 4. Vector Database

The generated embeddings are stored in a Chroma vector database.

When a user enters a query, the query is also converted into an embedding and compared with the stored book embeddings.

---

### 5. Semantic Retrieval

Chroma performs a similarity search and retrieves books whose descriptions are semantically similar to the user's query.

The system initially retrieves a larger set of potentially relevant books and then applies additional filtering.

---

### 6. Category and Emotional-Tone Filtering

Users can filter recommendations by category and emotional tone.

*Categories :*
- All
- Fiction
- Non Fiction
- Other available book categories
  
*Emotional tones :*
- Happy
- Sad
- Suspenseful
- Angry

The emotional scores are generated through emotion analysis and are used to rank books according to the selected tone.

---

### 7. Gradio Dashboard

The final recommendations are displayed through an interactive Gradio dashboard.

Each recommendation includes:

- 📖 Book cover
- 📕 Book title
- ✍️ Author
- 📝 Short description

---

### 🔄 Recommendation Pipeline

                User Query
                    │
                    ▼
        Sentence Transformer Model
                    │
                    ▼
             Query Embedding
                    │
                    ▼
          Chroma Similarity Search
                    │
                    ▼
           Similar Book Retrieval
                    │
                    ▼
        Category / Tone Filtering
                    │
                    ▼
          Ranked Recommendations
                    │
                    ▼
             Gradio Dashboard

---

### 💡 Example Queries

Try queries such as:
A story about forgiveness

A suspenseful adventure with unexpected twists

A book about personal growth and overcoming difficulties

A historical story about war

A funny story about friendship

A story about love and relationships

---

### 📂 Project Structure

semantic-book-recommender/
│
├── Gradio-dashboard.py
├── book-recommender.ipynb
├── books_cleaned.csv
├── books_with_emotion.csv
├── tagged_description.txt
├── cover_not_found.jpg
├── requirements.txt
├── .gitignore
└── README.md

### ⚙️ Installation
1. Clone the repository

```text
git clone <your-repository-url>
cd semantic-book-recommender
```
2. Install dependencies
```text
pip install -r requirements.txt
```
If required, install the Hugging Face embedding dependencies separately:
```text
pip install langchain-huggingface sentence-transformers
```

---

### 🙌 Acknowledgements

This project was inspired by the **LLM Course** – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio) tutorial by **Free Code Camp** (
freeCodeCamp.org)

The project was independently implemented and adapted, including the use of a local Hugging Face sentence-transformer embedding model, semantic search with Chroma, category classification, emotional-tone filtering, and the Gradio dashboard.

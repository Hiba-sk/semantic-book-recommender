print("SCRIPT STARTED")

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotion.csv")
books["large_thumbnail"] = books["thumbnail"] + "&file=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.jpg",
    books["large_thumbnail"],
)

raw_document = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_document)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)

db_books = Chroma.from_documents(documents, embeddings)
def retrieve_semantic_recommender(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [
        int(rec[0].page_content.strip('"').split()[0])
        for rec in recs
    ]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "ALL":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs= books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sad", ascending=False, inplace=True)

    return books_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommender(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_decription = " ".join(truncated_desc_split[:30]) + "...."

        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            author_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            author_str = f"{','.join(author_split[:-1])}, and {author_split[-1]}"
        else:
            author_str = row["authors"]

        caption = f"{row['title']} by {author_str}: {truncated_decription}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["ALL"] + sorted(books["simple_categories"].unique())
tones = ["ALL"] + ["Happy", "Sad", "Suspenseful", "Angry"]

with gr.Blocks(theme=gr.themes.Citrus()) as dashboard:
    gr.Markdown("# Semantic book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label= "Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="ALL")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select a emotional tone:", value="ALL")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("# Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
print("SCRIPT FINISHED")





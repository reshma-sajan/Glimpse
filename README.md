# 👁️ Glimpse

**A visual similarity & trend engine powered by CLIP.**

Upload any image. Glimpse tells you what else looks like it — and whether that aesthetic is rising or fading. One engine, four domains: fashion, art, food, and design.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- TODO: Add a demo GIF here once the Streamlit app is live -->
<!-- ![Glimpse Demo](assets/demo.gif) -->

---

## How It Works

```
Image → CLIP Embedding → FAISS Similarity Search → Top-K Matches
                      ↘ UMAP + HDBSCAN Clustering → Trend Analysis
```

1. **Embed** — Every image is passed through OpenAI's CLIP (ViT-B/32) to produce a 512-dimensional vector capturing its visual and semantic content.
2. **Search** — FAISS indexes these vectors for fast nearest-neighbour lookup. Upload an image, get back the most visually similar items in milliseconds.
3. **Cluster** — UMAP reduces the embedding space to 2D, and HDBSCAN finds natural aesthetic groupings without needing predefined labels.
4. **Trend** — By tracking cluster sizes and composition over time (or across collections), Glimpse surfaces which visual styles are emerging or declining.

---

## Domains

| Domain | Dataset | What It Shows |
|--------|---------|---------------|
| 🧥 **Fashion** | [Fashion Product Images (Kaggle)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) | Find similar clothing, detect style clusters, spot trending aesthetics |
| 🎨 **Art** | [WikiArt](https://www.kaggle.com/datasets/ipythonx/wikiart) | Find stylistically similar artworks across eras and movements |
| 🍜 **Food** | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) | Match dish presentations, surface plating and cuisine trends |
| 🎨 **Design** | [Unsplash Lite](https://unsplash.com/data) | Map visual aesthetics, find complementary styles for moodboards |

---

## Project Structure

```
glimpse/
├── glimpse/                # Core library
│   ├── __init__.py
│   ├── embedder.py         # CLIP embedding pipeline
│   ├── index.py            # FAISS index build & query
│   ├── cluster.py          # UMAP + HDBSCAN clustering
│   └── trends.py           # Trend scoring logic
├── notebooks/              # Exploration & domain demos
│   ├── 01_fashion.ipynb
│   ├── 02_art.ipynb
│   ├── 03_food.ipynb
│   └── 04_design.ipynb
├── app/
│   └── streamlit_app.py    # Interactive demo
├── data/                   # Dataset scripts (not raw data)
│   └── download.sh
├── assets/                 # Screenshots, demo GIFs
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/reshma-sajan/glimpse.git
cd glimpse

# Install dependencies
pip install -r requirements.txt

# Download a dataset (e.g., Fashion)
bash data/download.sh fashion

# Build the index
python -m glimpse.index --domain fashion

# Launch the app
streamlit run app/streamlit_app.py
```

---

## Tech Stack

- **[OpenAI CLIP](https://github.com/openai/CLIP)** (ViT-B/32) — image & text embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** — fast vector similarity search
- **[UMAP](https://umap-learn.readthedocs.io/)** — dimensionality reduction for visualisation
- **[HDBSCAN](https://hdbscan.readthedocs.io/)** — density-based clustering
- **[Streamlit](https://streamlit.io/)** — interactive web app
- **Python 3.10+**, NumPy, Pandas, Matplotlib, Plotly

---

## Roadmap

- [ ] Core embedding pipeline
- [ ] FAISS index with search API
- [ ] UMAP + HDBSCAN clustering
- [ ] Trend scoring module
- [ ] Fashion domain notebook
- [ ] Art domain notebook
- [ ] Food domain notebook
- [ ] Design domain notebook
- [ ] Streamlit app with image upload
- [ ] Deploy on HuggingFace Spaces
- [ ] Medium write-up

---

## Author

**Reshma Sara Sajan** — MSc Data Science, London School of Economics
- [GitHub](https://github.com/reshma-sajan)
- [Email](mailto:reshma.sara2002@gmail.com)

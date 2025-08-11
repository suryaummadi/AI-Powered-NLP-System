
---

# AI-Powered Multi-Task NLP System for Product Review Intelligence

This repository contains four modules implementing key NLP tasks to analyze product reviews, along with a Streamlit web app for real-time review analysis:

1. **Review Summarization**  
2. **Customer Experience Analytics**  
3. **Pros and Cons Extraction**  
4. **Quality Scoring**

Each module is provided as a Jupyter notebook (.ipynb) with code, explanations, and example outputs.

---

## Project Overview

This system leverages advanced NLP models like FLAN-T5, Gemini 2.5 Flash API, and RoBERTa — all available via Hugging Face — to extract structured insights from customer reviews. It aims to provide:

- Concise summaries of lengthy reviews  
- Analytics on customer experiences  
- Identification of pros and cons  
- Quality scoring based on review content

The notebooks include preprocessing, model inference, and post-processing pipelines for each task.

Additionally, the project includes a **Streamlit web app** (`app.py`) that provides an interactive interface for users to input reviews and view structured insights across all NLP components in real-time.

---

## Datasets

This project uses the following datasets from Hugging Face:

- The **Review Summarizer Dataset** is loaded directly from its CSV URL:  
  `hf://datasets/kartikay/review-summarizer/raw/data.csv`

- The **Product Review Sentiment** dataset is loaded via the Hugging Face datasets library using the identifier:  
  `Kenneth12/productreviewsentiment`

- The **Fake Reviews Dataset** is loaded via the Hugging Face datasets library using the identifier:  
  `theArijitDas/Fake-Reviews-Dataset`

All data loading, cleaning, and preprocessing steps are implemented within the notebooks.

---

## Getting Started

### Prerequisites

- Python 3.10.6  
- Jupyter Notebook or JupyterLab  

### Installation

1. Clone this repository:

```bash
git clone https://github.com/suryaummadi/AI-Powered-NLP-System.git
cd AI-Powered-NLP-System


2. Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open any of the four notebooks and run the cells sequentially.

---

## Running the Streamlit Web App

Run the interactive web app with:

```bash
streamlit run app.py
```

---

## Dependencies

The main libraries used include:

* transformers
* streamlit
* pandas
* scikit-learn
* numpy
* tensorflow
* datasets
* huggingface-hub

---

## Project Structure

```
AI-Powered-NLP-System/
│
├── Review_Summarization.ipynb  
├── Customer_Experience_Analytics.ipynb  
├── Pros_Cons_Extraction.ipynb  
├── Quality_Scoring.ipynb  
├── app.py                 # Streamlit web app  
├── README.md  
├── requirements.txt  
└── (other supporting files/folders)
```

---

## Author

Surya Venkata Sekhar Ummadi  
[GitHub Profile](https://github.com/suryaummadi)



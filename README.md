# 📝 Text Summarization Using Transformer Models

## 📌 Project Overview

**Text Summarization Using Transformer Models** is an NLP-based project developed using Python. It summarizes long text documents into short and meaningful summaries using Extractive and Abstractive Summarization techniques.

The project uses the **BART Transformer Model** to generate summaries and ROUGE scores to evaluate summary quality.

## 🎯 Objectives

* Automatically summarize lengthy text documents.
* Implement Extractive and Abstractive Summarization.
* Use a pre-trained BART model for summary generation.
* Evaluate summaries using ROUGE-1, ROUGE-2, and ROUGE-L.
* Reduce the time required to understand lengthy text.

## ✨ Features

* Automatic text summarization.
* Extractive summarization using TF-IDF.
* Abstractive summarization using BART.
* ROUGE-based summary evaluation.
* Comparison of summarization techniques.
* Visualization of evaluation results using charts.

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Notebook Environment:** Jupyter Notebook
* **NLP Model:** BART (`facebook/bart-large-cnn`)
* **Libraries:** Transformers, PyTorch, NLTK, Scikit-learn, Pandas, Matplotlib
* **Evaluation:** ROUGE Score

## ⚙️ Installation

Install the required libraries using the following command:

```bash
pip install transformers torch sentencepiece rouge-score nltk pandas matplotlib scikit-learn
```

## 🚀 How to Run

1. Clone or download this repository.
2. Open the project notebook in Jupyter Notebook.
3. Install the required Python libraries.
4. Run the notebook cells in sequence.
5. Provide the input text.
6. View the extractive and abstractive summaries.
7. Compare the results using ROUGE scores and charts.

## 📊 Methodology

1. **Text Input:** Accept the text document.
2. **Preprocessing:** Clean and prepare the input text.
3. **Extractive Summarization:** Select important sentences using TF-IDF and similarity scores.
4. **Abstractive Summarization:** Generate a new summary using the pre-trained BART model.
5. **Evaluation:** Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.
6. **Comparison:** Display and compare the summarization results.

## 🔮 Future Scope

* Support PDF and Word document uploads.
* Add Marathi, Hindi, and other languages.
* Develop a web-based application.
* Improve summarization accuracy.
* Support large documents and research papers.
* Add speech-to-text summarization.

## 🎓 Project Purpose

This project is developed for educational purposes to understand Natural Language Processing, transformer models, and automatic text summarization.

## 👨‍💻 Developed By

**Sanket Vijay Sahane**

## 📄 License

This project is intended for educational and learning purposes.

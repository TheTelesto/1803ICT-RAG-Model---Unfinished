# University Coursework Assistant - RAG Model

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) based model designed to assist with university coursework, specifically tailored to one unit. The goal was to create a tool that could help me with assignments by leveraging the power of AI to search and generate responses from my course materials. It is designed to streamline study sessions and improve the efficiency of working through university content.

## Motivation
I built this project to give myself a leg up on university work. Standard AI models are great, but they aren't specifically tailored to coursework, making this custom solution much more efficient. By focusing on a single unit, the model is fine-tuned to the needs of that specific subject. Originally, I planned to fine-tune a larger model, but due to the limited amount of available data, I decided to use a pre-trained model with retrieval-based enhancements. In the future, I’d like to revisit this idea and try fine-tuning on a smaller model to better fit the content.

## Installation and Setup
To get the project up and running, follow these steps:

### 1. Download Ollama
Ollama is the model interface required for RAG integration.

### 2. Install Python Libraries
You will need the following libraries installed to use the project:

- Faiss
- Numpy
- Requests
- Transformers
- JSON
- Langchain
- Flask
- You can install them using the command:

 ```bash
   pip install faiss numpy requests transformers json langchain flask
```
## Run the Model
The primary script for running the model is RAGModelWorking. Once everything is set up, simply run this file to get started.

Project Structure
RAGModelWorking.py: This is the core script for using the RAG model with your course material.
Course Material (Optional): This is the unit-specific content that will be used to assist with the retrieval and generation process.
## If I Had More Time
If I had more time, I would have focused on fine-tuning a smaller model instead of relying entirely on the retrieval-based system. Fine-tuning could make the model even more accurate for generating answers specific to the unit content. Another area for improvement would be expanding the dataset or finding a way to automate data extraction from additional course materials.

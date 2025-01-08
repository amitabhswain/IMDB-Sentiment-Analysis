# IMDB-Sentiment-Analysis

This is a Simple RNN project for sentiment analysis of IMDb movie reviews. The project demonstrates end-to-end deep learning implementation using TensorFlow and Keras, with deployment via Streamlit.

# Project Architecture
• Data ingestion of IMDb movie reviews dataset (50,000 records)
• Feature engineering and transformation using embedding layers
• Simple RNN model with embedding layer for text classification
• Streamlit web application for real-time predictions
• Cloud deployment capabilities

# Technical Components

**Model Architecture**

• Embedding layer for word vector representation
• Simple RNN layer with ReLU activation
• Dense output layer with sigmoid activation for binary classification
• Early stopping callback for optimal training


**Implementation Features**

• Text preprocessing with padding sequences
• Word embedding with 128-dimensional vectors
• Model training achieving ~94% training accuracy and ~81% validation accuracy
• Model persistence using H5 format
• Interactive web interface for real-time predictions


# Development Process
1. Data preprocessing and feature engineering
2. Model training with embedding layers
3. Model evaluation and persistence
4. Streamlit web application development
5. Cloud deployment setup


The project serves as a practical implementation of RNN architecture for sentiment analysis, combining deep learning concepts with production-ready deployment capabilities.

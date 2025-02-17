# Product Color Matching with FAISS and Cosine Similarity

## Overview

This project is designed to match clothing products based on their dominant colors using the CIELAB color space. By analyzing the colors of new product images and comparing them to an existing dataset of products, the system recommends similar products based on color similarity. The system utilizes FAISS for fast similarity searches and employs cosine similarity to improve the quality of matching results.

## Features

**Color Detection:** Extracts dominant colors from product images.

**CIELAB Color Space:** Colors are represented in the CIELAB color model for better perceptual uniformity.

**Similarity Matching:** Matches product colors using cosine similarity for accurate results.

**FAISS Indexing:** Uses FAISS for fast and efficient similarity searches over large datasets.

**Category Filtering:** Filters products by category to ensure relevant recommendations.

**Multi-color Matching:** Supports products with one, two, or three dominant colors.

## Technologies

Python 3.x

FAISS: Library for efficient similarity search and clustering of high-dimensional vectors.

NumPy: For numerical operations and vector manipulations.

OpenCV or Pillow: For image processing and color extraction.

OpenAI.

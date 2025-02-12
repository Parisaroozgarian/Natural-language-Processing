# Natural Language Processing - Document Classifier

## Description
This project implements a perceptron-based document classifier using scikit-learn. The classifier is trained on the 20 newsgroups dataset, specifically focusing on two categories: 'alt.atheism' and 'sci.med'. It uses TF-IDF (Term Frequency-Inverse Document Frequency) features for document representation.

## Features
- Document classification using Perceptron algorithm
- TF-IDF feature extraction
- Binary classification between religious and medical texts
- Pre-processing using scikit-learn's built-in tools

## Dependencies
- scikit-learn
- Python 3.x

## Usage
The classifier can be used to categorize text documents into two categories:
1. Religious content (alt.atheism)
2. Medical content (sci.med)

## Example Output
The classifier categorizes test documents like:
- "Religion is widespread, even in modern times" → Religious category
- "White blood cells fight off infections" → Medical category

## Installation

## License
MIT License
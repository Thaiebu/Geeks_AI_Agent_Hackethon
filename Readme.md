# AI Agent with LangGraph & Gemini API

## Overview
This project is a multi-agent AI system built using **LangGraph** and **Gemini API**, designed to deliver **hyper-personalized product recommendations** for an e-commerce platform. The system consists of two core tools:

- **Segmentize**: For automatic customer segmentation.
- **Recommendations**: For tailored product suggestions.

## Features

### ✨ AI Agent with LangGraph & Gemini API
- Developed a multi-agent framework using LangGraph.
- Integrated Gemini API to power intelligent interactions.
- Two agents/tools: `Segmentize` and `Recommendations`.

### 📅 Input Customer Details
The system collects and processes the following customer data:
- `Customer_ID`
- `Age`
- `Gender`
- `Location`
- `Browsing_History`
- `Purchase_History`
- `Avg_Order_Value`
- `Holiday`
- `Season`

### 🌐 Customer Segmentation (Segmentize Tool)
- Utilizes a pre-trained clustering model (`segmenter_model.joblib`).
- Automatically assigns customers into relevant segments based on their behavior and purchase history.

### 🛒 Product Recommendations (Recommendations Tool)
- Based on the customer's segment and details, recommends top products.
- Personalized suggestions boost engagement and conversion.

## Technologies Used
- Python
- Pandas, NumPy, Scikit-learn
- LangGraph
- Google Gemini API
- SentenceTransformers
- Pydantic
- Streamlit

## Project Structure
```
├── Readme.md
├── recommendation_agent.py         # Logic for recommendations
├── requirment.txt                  # Project dependencies
├── segmenter_model.joblib          # Trained ML model for segmentation
├── stream_ui.py                    # UI built with Streamlit
```

## How It Works
1. User provides input through a Streamlit UI.
2. The Segmentize agent predicts the customer segment.
3. The Recommendation agent uses this segment to suggest the top 3 products.
4. Results are displayed back in the UI with customer and product details.

## Conclusion
This project solves the challenge of delivering scalable, hyper-personalized product recommendations in e-commerce. With the power of LangGraph and Gemini, we automate customer insights and enhance engagement efficiently.

---

Feel free to fork, clone, or contribute to make this solution even more powerful!


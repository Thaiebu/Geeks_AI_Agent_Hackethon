import pandas as pd
import numpy as np
import ast
import os
import joblib
import google.generativeai as genai
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv


# Configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class RecommendationEngine:
    def __init__(self, products_df: pd.DataFrame):
        self.products = products_df
        self.location_mapping = {  # Map cities to countries
            'Chennai': 'India',
            'Delhi': 'India',
            'Bangalore': 'India',
            'Kolkata': 'India',
            'Mumbai': 'India'
        }
        
        # Define recommendation rules for each segment
        self.segment_rules = {
            'New Visitor': {
                'sort_columns': ['Probability_of_Recommendation', 'Product_Rating'],
                'sort_ascending': [False, False],
                'filters': [
                    ('Price', '<=', 2500),  # Focus on affordable items
                    ('Customer_Review_Sentiment_Score', '>=', 0.5)
                ]
            },
            'Occasional Shopper': {
                'sort_columns': ['Average_Rating_of_Similar_Products', 'Price'],
                'sort_ascending': [False, True],
                'filters': [
                    ('Product_Rating', '>=', 3.0),
                    ('Customer_Review_Sentiment_Score', '>=', 0.6)
                ]
            },
            'Frequent Buyer': {
                'sort_columns': ['Price', 'Probability_of_Recommendation'],
                'sort_ascending': [False, False],  # Premium items first
                'filters': [
                    ('Product_Rating', '>=', 4.0),
                    ('Price', '>=', 1500)
                ]
            }
        }
    
    def _map_location(self, customer_location: str) -> str:
        return self.location_mapping.get(customer_location, 'India')
    
    def get_recommendations(self, 
                           customer_data: Dict, 
                           segment: str, 
                           top_n: int = 5) -> List[Dict]:
        # Validate input
        if segment not in self.segment_rules:
            raise ValueError(f"Invalid segment: {segment}")
            
        # Get customer context
        country = self._map_location(customer_data['Location'])
        season = customer_data['Season']
        holiday = customer_data['Holiday']
        
        # Filter products by context
        filtered = self.products[
            (self.products['Geographical_Location'] == country) &
            (self.products['Season'] == season) &
            (self.products['Holiday'] == holiday)
        ]
        
        # Apply segment-specific filters
        rules = self.segment_rules[segment]
        for column, op, value in rules['filters']:
            if op == '<=':
                filtered = filtered[filtered[column] <= value]
            elif op == '>=':
                filtered = filtered[filtered[column] >= value]
            # Add more operators as needed
        
        # Sort products
        sorted_products = filtered.sort_values(
            by=rules['sort_columns'],
            ascending=rules['sort_ascending']
        )
        
        # Get top N recommendations
        recommendations = sorted_products.head(top_n)
        
        # Convert to list of dicts
        return recommendations.to_dict('records')
    


class CustomerSegmenter:
    def __init__(self, model_path):
        components = joblib.load(model_path)
        self.pipeline = components['pipeline']
        self.bert_model = components['bert_model']
        self.cluster_map = components['cluster_map']
        
    def preprocess_input(self, data):
        # Convert lists if they're strings
        if isinstance(data['Browsing_History'], str):
            data['Browsing_History'] = ast.literal_eval(data['Browsing_History'])
        if isinstance(data['Purchase_History'], str):
            data['Purchase_History'] = ast.literal_eval(data['Purchase_History'])
            
        return data
    
    def predict(self, input_data):
        # Preprocess
        input_data = self.preprocess_input(input_data)
        
        # Generate embeddings
        browsing_str = ', '.join(input_data['Browsing_History'])
        purchase_str = ', '.join(input_data['Purchase_History'])
        
        browsing_emb = self.bert_model.encode([browsing_str])
        purchase_emb = self.bert_model.encode([purchase_str])
        text_emb = np.hstack((browsing_emb, purchase_emb))
        
        # Transform through pipeline
        text_emb_pca = self.pipeline.named_steps['pca'].transform(text_emb)
        text_emb_scaled = self.pipeline.named_steps['text_scaler'].transform(text_emb_pca)
        
        numeric_features = np.array([
            input_data['Age'], 
            input_data['Avg_Order_Value']
        ]).reshape(1, -1)
        numeric_scaled = self.pipeline.named_steps['numeric_scaler'].transform(numeric_features)
        
        combined = np.hstack((text_emb_scaled, numeric_scaled))
        
        # Predict
        cluster = self.pipeline.named_steps['kmeans'].predict(combined)[0]
        return self.cluster_map[cluster]
    


# Agent State Model
class AgentState(BaseModel):
    customer_data: Optional[Dict[str, Any]] = None
    segment: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    next_step: str = "segmentize"
    final_response: Optional[str] = None

# Initialize your existing tools
segmenter = CustomerSegmenter('segmenter_model.joblib')
# Assuming products_df is available or loaded elsewhere
products_df = pd.read_csv('product_recommendation_data.csv')
engine = RecommendationEngine(products_df)

# Define tool functions that use your existing code
def segmentize_customer(state: AgentState) -> AgentState:
    """Use existing CustomerSegmenter to segment the customer."""
    customer_data = {
        'Age': state.customer_data.get('Age'),
        'Avg_Order_Value': state.customer_data.get('Avg_Order_Value'),
        'Browsing_History': state.customer_data.get('Browsing_History'),
        'Purchase_History': state.customer_data.get('Purchase_History')
    }
    
    # Use your existing segmenter to make prediction
    segment = segmenter.predict(customer_data)
    
    # Update state
    state.segment = segment
    state.next_step = "recommend"
    return state

def recommend_products(state: AgentState) -> AgentState:
    """Use existing RecommendationEngine to get product recommendations."""
    customer_data = {
        'Customer_ID': state.customer_data.get('Customer_ID'),
        'Location': state.customer_data.get('Location'),
        'Season': state.customer_data.get('Season'),
        'Holiday': state.customer_data.get('Holiday'),
        'Segment': state.segment
    }
    
    # Use your existing recommendation engine
    recommendations = engine.get_recommendations(
        customer_data=customer_data,
        segment=state.segment,
        top_n=3
    )
    
    # Update state
    state.recommendations = recommendations
    state.next_step = "summarize"
    return state

def generate_response(state: AgentState) -> AgentState:
    """Use Gemini to generate a personalized response with recommendations."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    segment = state.segment
    recommendations = state.recommendations
    customer_data = state.customer_data
    
    # Format recommendations for display
    recommendation_text = ""
    for product in recommendations:
        recommendation_text += f"- {product['Product_ID']}: {product['Category']} - {product['Subcategory']} (₹{product['Price']})\n"
    
    prompt = f"""
    Generate a personalized product recommendation message for a customer with the following details:
    - ID: {customer_data.get('Customer_ID')}
    - Age: {customer_data.get('Age')}
    - Gender: {customer_data.get('Gender')}
    - Location: {customer_data.get('Location')}
    - Season: {customer_data.get('Season')}
    - Customer Segment: {segment}
    
    Recommended Products:
    {recommendation_text}
    
    The response should be friendly, personalized, and explain 
    why these products are being recommended based on their segment and other factors.

    IMPORTANT: Use "Tata CLiQ" as the brand name throughout the message. 
    Do not use placeholders like [Your Brand Name].
    """
    
    response = model.generate_content(prompt)
    state.final_response = response.text
    state.next_step = "end"
    return state

# Define the LangGraph workflow
def build_recommendation_graph():
    # Create a new graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("segmentize", segmentize_customer)
    graph.add_node("recommend", recommend_products)
    graph.add_node("summarize", generate_response)
    
    # Add edges - define the workflow
    graph.add_edge("segmentize", "recommend")
    graph.add_edge("recommend", "summarize")
    graph.add_edge("summarize", END)
    
    # Set entry point
    graph.set_entry_point("segmentize")
    
    return graph.compile()


# Create the agent class
class RecommendationAgent:
    def __init__(self):
        self.workflow = build_recommendation_graph()
        
    def process_customer(self, customer_data: Dict[str, Any]) -> str:
        """Process customer data and return personalized recommendations."""
        # Initialize state
        initial_state = AgentState(customer_data=customer_data)
        
        # Execute the graph
        result = self.workflow.invoke(initial_state)
        
        return result
    
# # Example usage
# if __name__ == "__main__":
#     # Sample customer data for testing
#     sample_customer = {
#         'Customer_ID': 'C1234',
#         'Age': 32,
#         'Gender': 'Female',
#         'Location': 'Chennai',
#         'Browsing_History': ['Electronics', 'Fashion'],
#         'Purchase_History': ['Smartphone', 'Jeans'],
#         'Avg_Order_Value': 3500.00,
#         'Holiday': 'No',
#         'Season': 'Winter'
#     }
    
#     # Initialize and run the agent
#     agent = RecommendationAgent()
#     response = agent.process_customer(sample_customer)
#     print(response)
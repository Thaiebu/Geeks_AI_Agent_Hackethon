import streamlit as st
import pandas as pd
import os
from typing import Dict, Any
import json

# Import your recommendation agent (assuming it's in a file called recommendation_agent.py)
# If you have a different structure, adjust this import
from recommendation_agent import RecommendationAgent, CustomerSegmenter, RecommendationEngine

# Set page configuration
st.set_page_config(
    page_title="Tata CLiQ Customer Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# Initialize your tools and agent
@st.cache_resource
def load_models_and_data():
    """Load models and data that should be cached between sessions"""
    # Load product data - in production this might come from a database
    products_df = pd.read_csv("Dataset\product_recommendation_data.csv")  # Replace with your actual data source
    
    # Initialize segmenter and recommendation engine
    segmenter = CustomerSegmenter('segmenter_model.joblib')
    engine = RecommendationEngine(products_df)
    
    # Create the agent
    agent = RecommendationAgent()
    
    return segmenter, engine, agent

# Function to create a mock product dataframe for demo purposes
# Remove this in production and use your actual data
def create_mock_products():
    products = [
        {
            'Product_ID': 'P001', 
            'Category': 'Electronics', 
            'Subcategory': 'Smartphones',
            'Price': 15000.00,
            'Rating': 4.5,
            'Discount': 10,
            'Target_Locations': ['Chennai', 'Bangalore', 'Mumbai'],
            'Target_Seasons': ['Summer', 'Winter'],
            'Holiday_Special': False
        },
        {
            'Product_ID': 'P002', 
            'Category': 'Electronics', 
            'Subcategory': 'Laptops',
            'Price': 45000.00,
            'Rating': 4.8,
            'Discount': 5,
            'Target_Locations': ['Chennai', 'Delhi', 'Mumbai'],
            'Target_Seasons': ['All'],
            'Holiday_Special': True
        },
        {
            'Product_ID': 'P003', 
            'Category': 'Fashion', 
            'Subcategory': 'Jeans',
            'Price': 2000.00,
            'Rating': 4.2,
            'Discount': 20,
            'Target_Locations': ['All'],
            'Target_Seasons': ['Winter', 'Monsoon'],
            'Holiday_Special': False
        },
        {
            'Product_ID': 'P004', 
            'Category': 'Fashion', 
            'Subcategory': 'T-shirts',
            'Price': 800.00,
            'Rating': 4.0,
            'Discount': 15,
            'Target_Locations': ['Chennai', 'Bangalore', 'Hyderabad'],
            'Target_Seasons': ['Summer'],
            'Holiday_Special': False
        },
        {
            'Product_ID': 'P005', 
            'Category': 'Home', 
            'Subcategory': 'Furniture',
            'Price': 25000.00,
            'Rating': 4.6,
            'Discount': 8,
            'Target_Locations': ['All'],
            'Target_Seasons': ['All'],
            'Holiday_Special': True
        }
    ]
    df = pd.DataFrame(products)
    
    # Convert lists to strings for CSV storage
    for col in ['Target_Locations', 'Target_Seasons']:
        df[col] = df[col].apply(lambda x: json.dumps(x))
        
    return df

# Create mock data file if needed - remove in production
if not os.path.exists("Dataset\product_recommendation_data.csv"):
    mock_df = create_mock_products()
    mock_df.to_csv("products.csv", index=False)

# Try to load models and agent
try:
    segmenter, engine, agent = load_models_and_data()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# Main app
def main():
    # Header
    st.title("Tata CLiQ Customer Recommendation System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Customer Input", "About"])
    
    if page == "Customer Input":
        customer_input_page()
    else:
        about_page()

def customer_input_page():
    st.header("Customer Details")
    st.write("Enter customer information to generate personalized recommendations")
    
    # Create a form for customer details
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID", value="C1234")
            age = st.number_input("Age", min_value=18, max_value=100, value=32)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            location = st.selectbox("Location", ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata"])
        
        with col2:
            avg_order_value = st.number_input("Average Order Value (₹)", min_value=0.0, value=3500.0, step=500.0)
            holiday = st.selectbox("Holiday Period?", ["No", "Yes"])
            season = st.selectbox("Current Season", ["Summer", "Winter", "Monsoon", "Spring"])
        
        # Multiselect for browsing and purchase history
        st.subheader("Browsing History")
        browsing_history = st.multiselect(
            "Select categories browsed",
            ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books", "Toys"],
            default=["Electronics", "Fashion"]
        )
        
        st.subheader("Purchase History")
        purchase_history = st.multiselect(
            "Select products purchased",
            ["Smartphone", "Laptop", "Jeans", "Shirt", "Furniture", "Watch", "Shoes"],
            default=["Smartphone", "Jeans"]
        )
        
        # Submit button
        submitted = st.form_submit_button("Generate Recommendations")
    
    # Process the form if submitted
    if submitted:
        if not models_loaded:
            st.error("Models are not properly loaded. Please check the logs for errors.")
            return
            
        # Prepare customer data
        customer_data = {
            'Customer_ID': customer_id,
            'Age': age,
            'Gender': gender,
            'Location': location,
            'Browsing_History': browsing_history,
            'Purchase_History': purchase_history,
            'Avg_Order_Value': avg_order_value,
            'Holiday': holiday,
            'Season': season
        }
        
        # Show a spinner while processing
        with st.spinner("Generating personalized recommendations..."):
            try:
                # Process customer data through the agent
                response = agent.process_customer(customer_data)
                
                # Display results
                st.success("Recommendations generated successfully!")
                
                # Create two columns for the results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # st.subheader("Personalized Recommendations")
                    # st.markdown(response)
                    # Display options for viewing the response
                    st.subheader("Personalized Recommendations (Visual Format)")
                    try:
                        parsed = json.loads(response) if isinstance(response, str) else response

                        st.markdown(f"### Hello {parsed['customer_data']['Customer_ID']} 👋")
                        st.write(f"**Segment**: {parsed.get('segment', 'Unknown')}")
                        
                        st.markdown("### 🛍️ Top Recommendations")
                        for item in parsed.get("recommendations", []):
                            st.markdown(f"""
                            <div style="background-color:#f9f9f9;padding:10px;border-radius:10px;margin-bottom:10px">
                            <b>{item['Category']} > {item['Subcategory']}</b>  
                            - **Product ID**: {item['Product_ID']}  
                            - **Price**: ₹{item['Price']}  
                            - **Rating**: {item['Product_Rating']}  
                            - **Sentiment Score**: {item['Customer_Review_Sentiment_Score']}  
                            - **Recommendation Probability**: {item['Probability_of_Recommendation']}
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("### 📨 Final Email Response")
                        st.text_area("Email Template", value=parsed.get("final_response", ""), height=300)

                    except Exception as e:
                        st.error("Could not beautify the response.")
                with col2:
                    st.subheader("Customer Segment")
                    # Use the segmenter directly to show the segment
                    segment_data = {
                        'Age': age,
                        'Avg_Order_Value': avg_order_value,
                        'Browsing_History': browsing_history,
                        'Purchase_History': purchase_history
                    }
                    segment = segmenter.predict(segment_data)
                    
                    # Display segment with a nice badge-like style
                    st.markdown(f"""
                    <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; text-align:center;">
                        <h3 style="margin:0; color:#0073b7;">{segment}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display a simple explanation of the segment
                    segment_explanations = {
                        "High Value": "Customer who spends more than average and is likely to purchase premium products.",
                        "New Visitor": "Recent customer who is exploring the product range and building brand loyalty.",
                        "Price Sensitive": "Customer who responds well to discounts and looks for value.",
                        "Regular": "Consistent customer with moderate spending patterns."
                    }
                    
                    st.write(segment_explanations.get(segment, "Customer with unique shopping patterns."))
            
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.write("Please try again or contact support if the issue persists.")

def about_page():
    st.header("About this Application")
    
    st.write("""
    ## Tata CLiQ Customer Recommendation System
    
    This application uses machine learning and AI to provide personalized product recommendations 
    to Tata CLiQ customers. The system analyzes customer data including demographics, browsing 
    history, purchase patterns, and contextual information to suggest products that match 
    individual preferences and needs.
    
    ### How it Works
    
    1. **Customer Segmentation**: The system categorizes customers into segments based on their 
       behavior and demographics.
       
    2. **Product Matching**: Using the customer segment and other factors like location and 
       season, the system identifies products that are likely to appeal to the customer.
       
    3. **Personalized Messaging**: The Gemini AI model generates a personalized message explaining 
       the recommendations in a friendly, conversational way.
    
    ### Technologies Used
    
    - **LangGraph**: For orchestrating the workflow between different components
    - **Google Gemini API**: For generating natural language recommendations
    - **Machine Learning**: For customer segmentation and product matching
    - **Streamlit**: For the user interface
    
    ### Contact
    
    For support or questions, please contact the IT department.
    """)

if __name__ == "__main__":
    main()
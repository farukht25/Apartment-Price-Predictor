import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Set page config and custom CSS for a dark, sleek theme ---
st.set_page_config(
    page_title="Apartment Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# Custom CSS for a cooler look
st.markdown("""
<style>
    /* Main app container */
    .st-emotion-cache-18ni7ap {
        background-color: #0c151c;
        color: #e0e0e0;
    }
    .st-emotion-cache-16txtb3 {
        background-color: #0c151c;
    }

    /* Sidebar styling */
    .st-emotion-cache-13sdm0h {
        background-color: #1a232f;
        color: #e0e0e0;
    }

    /* Titles and text */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0;
    }
    
    /* Button styling */
    .st-emotion-cache-6q9sum {
        background-color: #2e8b57;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .st-emotion-cache-6q9sum:hover {
        background-color: #3cb371;
    }

    /* Metric styling */
    .st-emotion-cache-14u4k23 {
        border-left: 5px solid #3cb371;
    }

    /* Sliders styling */
    .st-emotion-cache-60b64d {
        background-color: #3b4252;
    }
    .st-emotion-cache-1v0s10r {
        color: white;
    }
    
    /* Selectbox styling */
    .st-emotion-cache-q8s20l {
        background-color: #3b4252;
    }
    
    /* Text input styling */
    .st-emotion-cache-13k941 {
        background-color: #3b4252;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("🏠 Apartment Price Predictor")
st.markdown("### Predict the value of an apartment based on its features.")
st.markdown("---")

# --- Load the saved model and feature names ---
try:
    final_model = joblib.load('final_real_estate_model_corrected.pkl')
    feature_names = joblib.load('feature_names_corrected.pkl')
    # Load the original DataFrame for context (e.g., median price)
    df_sample = joblib.load('df_sample.pkl')
except FileNotFoundError:
    st.error("Model or data files not found. Please ensure 'final_real_estate_model_corrected.pkl', 'feature_names_corrected.pkl', and 'df_sample.pkl' are in the same directory.")
    st.stop()


# --- Create Input Widgets using columns for a cleaner layout ---
st.sidebar.header("Property Features")
col1, col2 = st.sidebar.columns(2)

with col1:
    building_construction_year = st.slider("Building Construction Year", 1950, 2026, 2020)
    apartment_floor = st.slider("Apartment Floor", 1, 50, 2)
    apartment_bedrooms = st.slider("Number of Bedrooms", 1, 10, 2)
    apartment_bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
    
with col2:
    building_total_floors = st.slider("Total Building Floors", 1, 50, 5)
    apartment_rooms = st.slider("Number of Rooms", 1, 10, 3)
    apartment_total_area = st.slider("Apartment Total Area (m²)", 20, 1000, 120)
    apartment_living_area = st.slider("Apartment Living Area (m²)", 15, 900, 110)

country = st.sidebar.selectbox("Country", sorted([col.replace('country_', '') for col in feature_names if col.startswith('country_')]))

# --- Create a DataFrame from the user inputs ---
input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
input_data['building_construction_year'] = building_construction_year
input_data['building_total_floors'] = building_total_floors
input_data['apartment_floor'] = apartment_floor
input_data['apartment_rooms'] = apartment_rooms
input_data['apartment_bedrooms'] = apartment_bedrooms
input_data['apartment_bathrooms'] = apartment_bathrooms
input_data['apartment_total_area'] = apartment_total_area
input_data['apartment_living_area'] = apartment_living_area
current_year = 2024
input_data['building_age'] = current_year - building_construction_year
if input_data['building_age'][0] < 0:
    input_data['building_age'][0] = 0
input_data['total_rooms'] = apartment_bedrooms + apartment_bathrooms
if f'country_{country}' in input_data.columns:
    input_data[f'country_{country}'] = 1

# --- Prediction & Display with Better UI ---
st.markdown("---")
if st.button("Predict Price"):
    st.subheader("Prediction Results:")
    predicted_price = final_model.predict(input_data)[0]

    # Display the result using a Streamlit metric
    median_price = df_sample['price_in_USD'].median()
    st.metric(
        label="Predicted Price",
        value=f"${predicted_price:,.2f}",
        delta=f"vs. Median: ${predicted_price - median_price:,.2f}"
    )

    # --- Data Visualization Section ---
    st.markdown("### Feature Insights")
    
    # Predicted Price Distribution Chart
    st.markdown("#### Predicted Price vs. Market Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    ax_hist.hist(df_sample['price_in_USD'], bins=50, alpha=0.7, color='skyblue', label='Dataset Price Distribution')
    ax_hist.axvline(predicted_price, color='red', linestyle='dashed', linewidth=3, label='Your Prediction')
    ax_hist.set_title('Predicted Price vs. Market Distribution')
    ax_hist.set_xlabel('Price (USD)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()
    st.pyplot(fig_hist)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)

    with col3:
        st.write("**How Area Affects Price**")
        # Generate data for the plot
        area_range = np.linspace(df_sample['apartment_total_area'].min(), df_sample['apartment_total_area'].max(), 50)
        plot_data = input_data.copy()
        plot_data = pd.concat([plot_data] * 50, ignore_index=True)
        plot_data['apartment_total_area'] = area_range
        
        # Predict prices for the range
        prices_for_plot = final_model.predict(plot_data)
        
        fig, ax = plt.subplots()
        ax.plot(area_range, prices_for_plot)
        ax.scatter(input_data['apartment_total_area'], predicted_price, color='red', zorder=5)
        ax.set_title('Price vs. Total Area')
        ax.set_xlabel('Apartment Total Area (m²)')
        ax.set_ylabel('Predicted Price')
        st.pyplot(fig)

    with col4:
        st.write("**Top Feature Importances**")
        # Get feature importances from the model
        importances = final_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
        ax.set_title('Top 10 Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.gca().invert_yaxis()
        st.pyplot(fig)

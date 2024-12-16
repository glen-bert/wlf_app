import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Map column names to model file names
model_map = {
    "Tanaon": "tanaon_lstm_attention.keras",
    "Guihean": "guihean_lstm_attention.keras",
    "Upper Amusig": "upper_amusig_lstm_attention.keras",
    "Lower Amusig": "lower_amusig_lstm_attention.keras"
}

# Load a model based on column selection
@st.cache_resource
def load_trained_model(model_file):
    return load_model(model_file)

# Function to scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Function to make predictions
def make_predictions(model, scaled_data):
    predicted_level = []
    current_batch = scaled_data.reshape(1, 120, 1)

    for _ in range(24):  # Predicting 24 hours
        next_prediction = model.predict(current_batch, verbose=0)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_level.append(next_prediction)
    return predicted_level

# Streamlit App
st.title("24-Hour Water Level Forecasting")

st.write("""
Upload data (last 120 hours of Water Level values) as a CSV file. The model will predict the next 24-hour values using the corresponding model.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')

        # Select a parameter for prediction
        parameter = st.selectbox(
            "Select a parameter for prediction:",
            options=list(model_map.keys())
        )

        if parameter not in df.columns:
            st.error(f"The selected parameter '{parameter}' is not in the uploaded file.")
        else:
            # Extract the last 60 days of data for the selected parameter
            parameter_data = df[parameter].values[-120:]
            if len(parameter_data) != 120:
                st.error(f"The data must contain at least 60 rows for '{parameter}' to make predictions.")
            else:
                # Scale data
                scaled_data, scaler = scale_data(parameter_data)

                # Load the corresponding model
                model_file = model_map[parameter]
                model = load_trained_model(model_file)

                # Make predictions
                predictions = make_predictions(model, scaled_data)

                # Inverse scale predictions
                predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

                # Create a new time inder the next 24 hours
                last_time = pd.to_datetime(df['datetime'].values[-1])
                future_times = [last_time + pd.Timedelta(hours=i) for i in range (1,25)]

                # Display results
                st.write(f"### Predicted Levels for Next 24 Hours of {parameter}")
                prediction_df = pd.DataFrame({
                    'datetime': pd.to_datetime(future_times),
                    "Predicted Level": predicted_values
                })
                st.dataframe(prediction_df, width=800, height=400)

                #combine original dataframe with predictions
                df_combined = pd.concat([df, prediction_df[['datetime', 'Predicted Level']]]).reset_index(drop=True)
                df_combined.set_index('datetime', inplace=True)

                #Update line chart to include both historical and preducted values
                df_combined_ = df_combined.tail(120)
                st.write(f"### Actual for the last 120 hours and Predicted Water Level for Next 24 Hours of {parameter}")
               
                df_combined_ = df_combined.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_combined_.index, y=df_combined_[parameter], mode='lines', name=f'Historical {parameter}', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df_combined_.index, y=df_combined_['Predicted Level'], mode='lines', name='Predicted Level', line=dict(color='orange')))
                fig.update_layout(title=f'Water Level Forecasting for {parameter}'
                                  , xaxis_title='Datetime'
                                  , yaxis_title ='Water Level'
                                  , xaxis_tickangle=-45,
                                  legend=dict(
                                        yanchor="top",
                                        xanchor="right",
                                    ))

                # Display the plot in Streamlit
                st.plotly_chart(fig)
    

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to start.")
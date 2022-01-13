import streamlit as st

"""  
# Credit Card Fraud Detector - A Unique Approach"""

# Train the model
st.write(
    """ 
#### Realtime Training and Prediction
"""
)

# buttons for training

if st.sidebar.button("Train Model"):
    st.write("Training the model....")
    # train the model by running the main.py file
    import os

    os.system("python main.py")
    # button for downloading the logs
    if st.sidebar.button("Show Logs"):
        # show logs there 
        st.write("Show txt file") 
        st.markdown("[Download Logs](logs.txt)")  
        
    st.write("Model Trained")

import streamlit as st

# Create a form using the beta_form function
with st.form(key='my_form'):
    input_text = st.text_input(label='Enter some text')
    input_number = st.number_input(label='Enter a number', min_value=0, max_value=100, value=50)
    input_date = st.date_input(label='Enter a date')
    submit_button = st.form_submit_button(label='Submit')

# Display the input values after the form is submitted
if submit_button:
    st.write(f'You entered text: {input_text}')
    st.write(f'You entered number: {input_number}')
    st.write(f'You entered date: {input_date}')


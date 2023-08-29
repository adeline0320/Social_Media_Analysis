import streamlit as st

# Define the navigation items and their corresponding Python files
navigation_items = {
    "Cadbury": "cadbury.py",
    "Kitkat": "kidkat.py",
    "M&M": "mm.py",
    "Smarties": "smarties.py",
    "Snickers": "snickers.py"
}

st.sidebar.title("Chocolate Brand Analysis")

# Create the side navigation bar
nav_selection = st.sidebar.radio("", list(navigation_items.keys()))

selected_file = navigation_items[nav_selection]
exec(open(selected_file).read())

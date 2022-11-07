import streamlit as st
import toml
import boto3

# def main_page():
#     st.markdown("# Main page")
#     st.sidebar.markdown("# Main page")

# def single_doc_page():
#     st.markdown("# See a quick demo on a single document")
#     st.sidebar.markdown("# Single Document")

# def multi_docs_page():
#     st.markdown("# Upload multiple documents and have the results sent to you")
#     st.sidebar.markdown("# Multiple Documents")

# page_names_to_funcs = {
#     "Main Page": main_page,
#     "Single Document": single_doc_page,
#     "Multiple Documents": multi_docs_page
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()



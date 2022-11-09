from io import StringIO
import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
import json
import boto3
from time import perf_counter
import plotly.express as px
import toml
from utils import query_lambda
from botocore.config import Config
from gensim.summarization.summarizer import summarize
import re


secrets = toml.load('.streamlit/secrets.toml')
aws_access_key_id = secrets['aws_access_key_id']
aws_secret_access_key = secrets['aws_secret_access_key']

config = Config(
        connect_timeout=60, read_timeout=900,
        )

lambda_client = boto3.client('lambda', 
            region_name='ap-southeast-1', 
            aws_access_key_id=aws_access_key_id, 
            aws_secret_access_key=aws_secret_access_key,
            config=config
            )


def main():
    st.set_page_config(
        layout="wide",
        page_title="Document Tagging"
        )
    st.title("Document Tagging: Use your own labels!")


    with st.sidebar:
        with st.container():
            st.write('## Labels')
            # st.write('### File Upload')
            st.write('#### Upload label file (one label per line in .txt format):')
            uploaded_labels = st.file_uploader('Upload labels', type=['txt'], label_visibility='hidden')

            # st.write('### Manual Entry')
            
            input_labels = st_tags(
                label='#### Enter labels (will be ignored if label file is uploaded):',
                text='Press enter to add',
                value=[],
                suggestions=['Economy', 'Education', 'Politics', 
                'Current Affairs', 'Crime', 'Housing', 
                'Sustainability', 'Food']
            )

            multi_label_options = ['Document can have multiple labels', 'One dominant label per document']
            multi_label_selection = st.radio("Is your document multi-label",
                                    multi_label_options, label_visibility="hidden")
            multi_label = multi_label_options.index(multi_label_selection) == 0

    input_container = st.container()
    input_text = input_container.text_area("Enter Text", height=300)
    submit = st.button('Submit')

    with st.container():
        if submit:
            selected_labels = None
            if uploaded_labels is not None:
                label_text = StringIO(uploaded_labels.getvalue().decode("utf-8")).read()
                selected_labels = [l.strip() for l in label_text.split('\n')]
            elif input_labels is not None:
                selected_labels = input_labels

            if selected_labels is None:
                st.write("Please make sure that you have uploaded or entered your labels")
            elif not input_text.strip():
                st.write("Please enter text")
            else:            
                with st.spinner('Processing....'):
                    if len(re.split('/s+', input_text)) > 400:
                        st.write('Input text too long. Summarising.....')
                        input_text = summarize(input_text, word_count=400)
                    predictions = query_lambda(input_text, labels=selected_labels, multi_label=multi_label, lambda_client=lambda_client)
                    df_predictions = pd.DataFrame(predictions, columns=['Label', 'Score']).sort_values('Score', ascending=False)
                    fig=px.bar(df_predictions, y='Label',x='Score', orientation='h', color='Score')
                st.write(fig)
        
        



if __name__ == "__main__":
    main()
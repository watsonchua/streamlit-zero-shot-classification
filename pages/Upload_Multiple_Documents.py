from io import StringIO
import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
import json
import boto3
from time import perf_counter
import plotly.express as px
import toml
import os
from datetime import datetime
from botocore.config import Config
from utils import query_lambda
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

s3_client = boto3.client('s3', 
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='ap-southeast-1'
)


s3_bucket_name = "zero-shot-classification-bucket"
s3_bucket = boto3.resource('s3').Bucket(s3_bucket_name)

def main():
    st.set_page_config(
        layout="wide",
        page_title="Document Tagging"
        )
    st.title("Document Tagging: Use your own labels!")


    # label container

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




        # add_labels = st.button('Add labels from file')
        # if add_labels:
        #     if uploaded_labels is not None:
        #         label_text = StringIO(uploaded_labels.getvalue().decode("utf-8")).read()
        #         input_labels += [l.strip().encode() for l in label_text.split('\n')]

    


    with st.container():
        uploaded_docs = st.file_uploader("Upload documents (.txt files only)", type=['txt'], accept_multiple_files=True)
        submit = st.button('Submit')

        if submit:
            selected_labels = None
            if uploaded_labels is not None:
                label_text = StringIO(uploaded_labels.getvalue().decode("utf-8")).read()
                selected_labels = [l.strip() for l in label_text.split('\n')]
            elif input_labels is not None:
                selected_labels = input_labels

            
            
            if selected_labels is None:
                st.write("Please make sure that you have uploaded or entered your labels")
            elif uploaded_docs is None:
                st.write("Please upload one or more documents")
            else:
                results = []
                
                doc_progress = st.progress(0)

                for i,ud in enumerate(uploaded_docs):
                    print(ud)
                    doc_results = {}
                    input_text = StringIO(ud.getvalue().decode("utf-8")).read()
                    if len(re.split('/s+', input_text)) > 400:
                        st.write('Input text too long. Summarising.....')
                        summarized_text = summarize(input_text, word_count=400)
                    else:
                        summarized_text = input_text
                    predictions = query_lambda(summarized_text, labels=selected_labels, multi_label=multi_label, lambda_client=lambda_client)
                    # predictions = [('Economy', 0.5), ('Healthcare', 0.8)]
                    doc_results['filename'] = ud.name
                    doc_results['text'] = input_text
                    doc_results['summarized_text'] = summarized_text
                    doc_results.update({pred: score for pred,score in predictions})
                    doc_progress.progress((i+1)/len(uploaded_docs))
                    results.append(doc_results)

                
                df_results = pd.DataFrame(results)


                results_filename = 'prediction_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'
                
                
                with StringIO() as csv_buffer:
                    df_results.to_csv(csv_buffer, index=False)

                    response = s3_client.put_object(
                        Bucket=s3_bucket_name, Key=results_filename, Body=csv_buffer.getvalue()
                    )

                    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

                    s3_status_message = ""
                    
                    if status == 200:
                        s3_status_message = f"Results successfully uploaded to S3. Status - {status}"
                    else:
                        s3_status_message = f"Uploading of results to S3 unsuccessful. Status - {status}"
                
                
                    # save_df_to_s3(df_results, results_filename)

                    st.success('Predictions Done!' + '\n' + s3_status_message)
                    # st.write("[Download Results](%s)" % file_download_link(results_filename))
                    
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=results_filename,
                        mime='text/csv',
                    )



                    df_temp = df_results[selected_labels]
                    df_temp_mean = df_temp.mean().to_frame('Score').sort_values('Score', ascending=False)
                    st.write("Corpus labels (mean values)")
                    fig=px.bar(df_temp_mean, y=df_temp_mean.index,x='Score', orientation='h', color='Score')
                    st.write(fig)

    




if __name__ == "__main__":
    main()
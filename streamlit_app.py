import streamlit as st
from streamlit_tags import st_tags, st_tags_sidebar
import pandas as pd
import json
import boto3
from time import perf_counter
import plotly.express as px
import toml

secrets = toml.load('.streamlit/secrets.toml')
aws_access_key_id = secrets['aws_access_key_id']
aws_secret_access_key = secrets['aws_secret_access_key']

lambda_client = boto3.client('lambda', 
            region_name='ap-southeast-1', 
            aws_access_key_id=aws_access_key_id, 
            aws_secret_access_key=aws_secret_access_key
            )


# lambda function reads as input event with key 'body' and value stringnified version of query
# stringnified version of query is then passed through json.loads again
# refer to app.py 

def query_lambda(text, labels):
    query = {
        "sequence": text,
        "labels": labels
        }

    query_str = json.dumps(query)

    lambda_payload = {
            "body": query_str
        }

    lambda_payload_str = json.dumps(lambda_payload)


    start = perf_counter()
    lambda_response = lambda_client.invoke(
                        FunctionName="zero-shot-classification-function",
                        InvocationType='RequestResponse',
                        Payload=bytes(lambda_payload_str, "utf-8")
                        )
    end = perf_counter()
    print('time taken:', end-start)

    results = lambda_response['Payload'].read()
    body = json.loads(results)['body']
    predictions = json.loads(body)['predictions']

    return predictions



def main():
    st.set_page_config(
        layout="wide",
        page_title="Document Tagging"
        )
    st.title("Document Tagging: Use your own labels!")


    # label container

    input_labels = st_tags_sidebar(
        label='# Enter labels',
        text='Press enter to add',
        value=[],
        suggestions=['Economy', 'Education', 'Politics', 
        'Current Affairs', 'Crime', 'Housing', 
        'Sustainability', 'Food']
    )

    input_container = st.container()
    input_text = input_container.text_area("Enter Text", height=300)
    submit = st.button('Submit')

    with st.container():
        if submit:
            if input_text.strip() and input_labels:
                with st.spinner('Processing....'):
                    predictions = query_lambda(input_text, labels=input_labels)
                    df_predictions = pd.DataFrame(predictions, columns=['Label', 'Score']).sort_values('Score', ascending=False)
                    fig=px.bar(df_predictions, y='Label',x='Score', orientation='h', color='Score')
                st.write(fig)
            else:
                st.write("Please enter both text and labels.")
        
        



if __name__ == "__main__":
    main()
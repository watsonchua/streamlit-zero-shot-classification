import json
from time import perf_counter
from ftfy import fix_text


# lambda function reads as input event with key 'body' and value stringnified version of query
# stringnified version of query is then passed through json.loads again
# refer to app.py 

def query_lambda(text, labels, lambda_client):
    query = {
        "sequence": fix_text(text),
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
    print(results)
    body = json.loads(results)['body']
    predictions = json.loads(body)['predictions']

    return predictions
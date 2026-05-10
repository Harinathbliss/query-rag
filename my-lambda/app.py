import json


def lambda_handler(event,context):
    print("Lambda Invoked")
    return {
        "statusCode":200,
        "body":json.dumps({
            "message":"Lambda Invoked Successfully"
        })
    }

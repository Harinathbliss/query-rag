import json



def lambda_handler(event,context):
    print("Lambda Invoked")
    request_body = event.get('body') or {}
    user_id = request_body.get('userId')
    query = request_body.get('query')
    
    print("User Id",user_id)
    print("Query",query)
    
    return {
    "statusCode": 200,
    "body": json.dumps({"message": f"User Asked {query} for User Id {user_id}"})
    }

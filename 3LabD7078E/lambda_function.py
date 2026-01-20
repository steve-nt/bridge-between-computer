import boto3
import json

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Debug: Log the entire event
    print("DEBUG: Full event received:")
    print(json.dumps(event, indent=2, default=str))
    
    proxy = event.get('pathParameters', {}).get('proxy', '')
    bucket = 'bucket-d7078e-lab3-group35'
    
    print(f"DEBUG: proxy = '{proxy}'")
    print(f"DEBUG: bucket = '{bucket}'")
    
    try:
        response = s3.get_object(Bucket=bucket, Key=proxy)
        body = response['Body'].read().decode('utf-8')
        return {
            'statusCode': 200,
            'body': body
        }
    except Exception as e:
        print(f"DEBUG: Exception = {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

#!/usr/bin/python
"""
Lambda Function that applies a Trigger Event to the S3 Bucket. This funciton is used to the 
LNN CloudFormation depployments.
"""

# Import Libraries
import json
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
from botocore.vendored import requests
import uuid

SUCCESS = 'SUCCESS'
FAILED = 'FAILED'

def send(event, context, responseStatus, responseData, physicalResourceId):
    responseUrl = event['ResponseURL']
    responseBody = {}
    responseBody['Status'] = responseStatus
    responseBody['Reason'] = 'See the details in CloudWatch Log Stream: ' + context.log_stream_name
    responseBody['PhysicalResourceId'] = physicalResourceId
    responseBody['StackId'] = event['StackId']
    responseBody['RequestId'] = event['RequestId']
    responseBody['LogicalResourceId'] = event['LogicalResourceId']
    responseBody['Data'] = responseData
 
    json_responseBody = dumps(responseBody)
 
    print("Response body:\n" + json_responseBody)
 
    headers = {
        'content-type' : '',
        'content-length' : str(len(json_responseBody))
    }
 
    try:
        response = requests.put(responseUrl,
                                data=json_responseBody,
                                headers=headers)
        print("Status code: " + response.reason)
    except Exception as e:
        print("send(..) failed executing requests.put(..): " + str(e))

def lambda_handler(event, context):
    responseData = {'Data': 'OK'}
    print(dumps(event))

    properties = event['ResourceProperties']
    bucket_name = properties['Bucket']
    rgn = properties['Region']
    s3_client = client('s3', region_name=rgn)
    lambda_client = client('lambda', region_name=rgn)

    # Define notification configuration
    configuration = {}
    if event['RequestType'] != 'Delete':
        configuration['LambdaFunctionConfigurations'] = [
            {
                'LambdaFunctionArn': properties['FunctionArn'],
                'Events': [
                    's3:ObjectCreated:Put'
                ],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {
                                'Name': 'suffix',
                                'Value': 'h5'
                            },
                            {
                                'Name': 'prefix',
                                'Value': 'training_input/'
                            }
                        ]
                    }
                }
            }
        ]
        
        # Create Permission to trigger the LaunchLambda
        try: 
            lambda_response = lambda_client.add_permission(
                FunctionName=properties['FunctionArn'],
                StatementId = str(uuid.uuid4()),
                Action='lambda:InvokeFunction',
                Principal='s3.amazonaws.com',
                SourceArn='arn:aws:s3:::' + bucket_name,
                SourceAccount=properties['AccountNumber'],
            )
            print("Create Permission Reponse: \n" + str(lambda_response))
        except Exception as e:
            print("Error Creating Permission: \n" + str(e))
            return send(event, context, FAILED, responseData, "CustomResourcePhysicalID")
    
    # Create Notification
    try:
        s3_response = s3_client.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=configuration
        )
        print("Create S3 Notification Response: \n" + str(s3_response))
        return send(event, context, SUCCESS, responseData, "CustomResourcePhysicalID")
    except Exception as e:
        print("Error Creating S3 Notification: \n" + str(e))
        return send(event, context, FAILED, responseData, "CustomResourcePhysicalID")
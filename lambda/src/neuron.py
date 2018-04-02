#!/usr/bin/python
"""
Lambda Function that simulates a single Neuron for both forward and backward propogation.
If the state is `forward` then the function simulates forward propogation for `X` to the `Cost`.
If the state is backward, then the function calculates the gradient of the derivative of the 
activation function for the current layer.
"""

# Import Libraries needed by the Lambda Function
import sys
import numpy as np
import h5py
import scipy
import os
from os import environ
import json
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
import uuid
import io
import redis
from redis import StrictRedis as redis

# Global Variables
rgn = environ['Region']
s3_client = client('s3', region_name=rgn) # S3 access
s3_resource = resource('s3')
sns_client = client('sns', region_name=rgn) # SNS
redis_client = client('elasticache', region_name=rgn)
lambda_client = client('lambda', region_name=rgn) # Lambda invocations
# Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
cache = redis(host=endpoint, port=6379, db=0)
dynamo_client = client('dynamodb', region_name=rgn)
dynamo_resource = resource('dynamodb', region_name=rgn)

# Helper Functions
def inv_counter(name, invID, task):
    """
    Manages the Counter assigned to a unique Lambda Invocation ID, by
    either setting it to 0, updating it to 1 or querying the value.
   
    Arguments:
    name -- The Name of the function being invoked
    invID -- The unique invocation ID created for the specific invokation
    task -- Task to perfoirm: set | get | update
    """
    table = dynamo_resource.Table(name)
    if task == 'set':
        table.put_item(
            Item={
                'invID': invID,
                'cnt': 0
            }
        )
        
    elif task == 'get':
        task_response = table.get_item(
            Key={
                'invID': invID
            }
        )
        item = task_response['Item'] 
        return int(item['cnt'])
        
    elif task == 'update':
        task_response = table.update_item(
            Key={
                'invID': invID
            },
            UpdateExpression='SET cnt = :val1',
            ExpressionAttributeValues={
                ':val1': 1
            }
        )

def publish_sns(sns_message):
    """
    Publish message to the master SNS topic.

    Arguments:
    sns_message -- the Body of the SNS Message
    """

    print("Publishing message to SNS topic...")
    sns_client.publish(TargetArn=environ['SNSArn'], Message=sns_message)

def to_cache(endpoint, obj, name):
    """
    Serializes multiple data type to ElastiCache and returns
    the Key.
    
    Arguments:
    endpoint -- The ElastiCache endpoint
    obj -- the object to srialize. Can be of type:
            - Numpy Array
            - Python Dictionary
            - String
            - Integer
    name -- Name of the Key
    
    Returns:
    key -- For each type the key is made up of {name}|{type} and for
           the case of Numpy Arrays, the Length and Widtch of the 
           array are added to the Key.
    """
    
    # Test if the object to Serialize is a Numpy Array
    if 'numpy' in str(type(obj)):
        array_dtype = str(obj.dtype)
        if len(obj.shape) == 0:
            length = 0
            width = 0
        else:
            length, width = obj.shape
        # Convert the array to string
        val = obj.ravel().tostring()
        # Create a key from the name and necessary parameters from the array
        # i.e. {name}|{type}#{length}#{width}
        key = '{0}|{1}#{2}#{3}'.format(name, array_dtype, length, width)
        # Store the binary string to Redis
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    # Test if the object to serialize is a string
    elif type(obj) is str:
        key = '{0}|{1}'.format(name, 'string')
        val = obj
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    # Test if the object to serialize is an integer
    elif type(obj) is int:
        key = '{0}|{1}'.format(name, 'int')
        # Convert to a string
        val = str(obj)
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    # Test if the object to serialize is a dictionary
    elif type(obj) is dict:
        # Convert the dictionary to a String
        val = json.dumps(obj)
        key = '{0}|{1}'.format(name, 'json')
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    else:
        sns_message = "`to_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported serialization type")
        raise

def from_cache(endpoint, key):
    """
    De-serializes binary object from ElastiCache by reading
    the type of object from the name and converting it to
    the appropriate data type
    
    Arguments:
    endpoint -- ElastiCacheendpoint
    key -- Name of the Key to retrieve the object
    
    Returns:
    obj -- The object converted to specifed data type
    """
    
    # Check if the Key is for a Numpy array containing
    # `float64` data types
    if 'float64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        if int(length) == 0:
            obj = np.float64(np.fromstring(val))
        else:
            obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a Numpy array containing
    # `int64` data types
    elif 'int64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a json type
    elif 'json' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        obj = cache.get(key)
        return json.loads(obj)
    # Chec if the Key is an integer
    elif 'int' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        obj = cache.get(key)
        return int(obj)
    # Check if the Key is a string
    elif 'string' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        obj = cache.get(key)
        return obj
    else:
        sns_message = "`from_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported de-serialization type")
        raise

def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size

    Return:
    sigmoid(z)
    """
    return 1. / (1. + np.exp(-z))

def sigmoid_backward(dA, z):
    """
    Implement the derivative of the sigmoid function

    Arguments:
    dA -- Post-activation gradient, of any shape
    z -- Cached linear activation from Forward prop

    Returns:
    dZ -- Gradient of the Cost with respect to z
    """
    s = 1. / (1. + np.exp(-z))
    dZ = dA * s * (1 - s)
    # Debug statement
    #assert(dZ.shape == z.shape)
    return dZ

def relu(z):
    """
    Implement the ReLU function.

    Arguments:
    z -- Output of the linear layer, of any shape

    Returns:
    a -- Post-activation parameter, of the same shape as z
    """
    a = np.maximum(0, z)
    # Debug statement
    #assert(a.shape == z.shape)
    return a

def relu_backward(dA, z):
    """
    Implement the backward propagation for a single ReLU unit

    Arguments:
    dA -- Post-activation gradient, of any shape
    z -- Cached linear activation from Forward propagation

    Return:
    dz -- Gradient of the Cost with respect to z
    """
    dz = np.array(dA, copy=True) #converting dz to a correct object
    # When z <= 0, set dz to 0 as well
    dz[z <= 0] = 0
    # Debug statement
    #assert (dz.shape == z.shape)
    return dz

def leaky_relu(z):
    """
    Implement the Leaky ReLU function.

    Arguments:
    z -- Output of the linear layer, of any shape

    Returns:
    a -- Post-activation parameter, of the same shape as z
    """
    a = np.maximum(0,z)
    a[z < 0] = 0.01 * z
    return a

def leaky_relu_backward(dA, z):
    """
    Implement the backward propagation for a single Leaky ReLU unit.

    Argument:
    dA -- Post-activation gradient, of any shape
    z -- Cached linear activation from Forward propagation
    """
    dz = np.array(dA, copy=True)
    dz[z <= 0] = 0.01
    return dz

def lambda_handler(event, context):
    """
    This Lambda Funciton simulates a single Perceptron for both 
    forward and backward propogation.
    """
    # Ensure that this is not a duplicate invokation
    invID = event.get('invID')
    name = "NeuronLambda" #Name of the current Lambda function
    task = 'get'
    cnt = inv_counter(name, invID, task) #should be 0 for a new function invoked
    if cnt == 0:
        task = 'update'
        inv_counter(name, invID, task)
    else:
        sys.exit()
    
    # Get the Neural Network parameters from Elasticache
    parameters = from_cache(endpoint, key=event.get('parameter_key'))
       
    # Get the current state
    state = event.get('state')
    epoch = event.get('epoch')
    layer = event.get('layer')
    ID = event.get('id') # To be used when multiple activations
    # Determine is this is the last Neuron in the layer
    last = event.get('last')
    activation = event.get('activation')
    print("Starting {} propagation on Neuron: {}, for Epoch {} and Layer {}".format(state, str(ID), str(epoch), str(layer)))

    if state == 'forward':
        # Forward propogation from A0 to Cost
        # Activations from the previous layer
        A_prev = from_cache(endpoint=endpoint, key=parameters['data_keys']['A'+str(layer - 1)])
        # Get the weights for this neuron
        w = from_cache(endpoint=endpoint, key=parameters['data_keys']['W'+str(layer)])[ID-1, :]
        # Convert weights to a row vector
        w = w.reshape(1, w.shape[0])
        # Get the bias for this neuron as row vector
        b = from_cache(endpoint=endpoint, key=parameters['data_keys']['b'+str(layer)])[ID-1, :]
        # Perform the linear part of the layer's forward propogation
        z = np.dot(w, A_prev) + b
        # Upload the linear transformation results to ElastiCache for use with Backprop
        to_cache(endpoint=endpoint, obj=z, name='layer'+str(layer)+'_z_'+str(ID))

        # Perform non-linear activation based on the activation function
        if activation == 'sigmoid':
            a = sigmoid(z)
        elif activation == 'relu':
            a = relu(z)
        else:
            # No other functions supported at this time
            pass
        # Upload the results to ElastiCache for `TrainerLambda` to vectorize
        to_cache(endpoint=endpoint, obj=a, name='layer'+str(layer)+'_a_'+str(ID))

        print("Completed Forward Propogation for epoch {}, layer {}".format(str(epoch), str(layer)))
        
        if last == "True":
            # Update parameters with this Neuron's data
            parameters['epoch'] = epoch
            parameters['layer'] = layer + 1
            # Build the state payload
            payload = {}
            payload['parameter_key'] = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
            payload['state'] = 'forward'
            payload['epoch'] = epoch
            payload['layer'] = layer + 1

            # Crate an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "TrainerLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)

            # Debug Statement
            #print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))

            # Invoke TrainerLambda to process activations
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['TrainerLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking Trainer Lambd from NeuronLambdaa."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
            #print(response)

        return

    elif state == 'backward':
        # Backprop from Cost to X (A0)
        """
        Inutition: TrainerLambda launched back prop with `layer-1`, therefore this should be 
        last "active" layer. That means that the "dA" for this layer has already been
        calculate. Thus, no need to do the `A - Y` error calculation. Additionally, 
        the following code structure makes the it more idempotenent for multiple layers.
        """
        # Get necessary parameters
        r = redis(host=endpoint, port=6379, db=0, charset="utf-8", decode_responses=True)
        z_key = []
        for z in r.scan_iter(match='layer'+str(layer)+'_z_'+str(ID)+'|*'):
            z_key.append(z)
        z = from_cache(endpoint=endpoint, key=z_key[0])
        m = from_cache(endpoint=endpoint, key=parameters['data_keys']['m'])
        A_prev = from_cache(endpoint=endpoint, key=parameters['data_keys']['A'+str(layer-1)])

        # Get the derivative of the current layer's activation,
        # based on the size of the layer.
        if layer == parameters['layers']:
            # If this is the last layer, then:
            dA = from_cache(endpoint=endpoint, key=parameters['data_keys']['dA'+str(layer)])
            W = from_cache(endpoint=endpoint, key=parameters['data_keys']['W'+str(layer)])
        else:
            dA = from_cache(endpoint=endpoint, key=parameters['data_keys']['dA'+str(layer)])[ID-1, :]
            dA = dA.reshape(1, dA.shape[0])
            W = from_cache(endpoint=endpoint, key=parameters['data_keys']['W'+str(layer)])[ID-1, :]
            W = W.reshape(1, W.shape[0])
        
        # Calculate the derivative of the Activations
        if activation=='sigmoid':
            dZ = sigmoid_backward(dA, z)
        elif activation == 'relu':
            dZ = relu_backward(dA, z)
        elif activaion == 'leaky_relu':
            dZ = leaky_relu_backward(dA, z)
        else:
            # No other functions supported at this time
            pass
        # Upload the derivative of the activation to ElastiCache for use by `TrainerLambda`
        to_cache(endpoint=endpoint, obj=dZ, name='layer'+str(layer)+'_dZ_'+str(ID))
        
        # Calculate the derivatives of the weights
        dw = 1 / m * np.dot(dZ, A_prev.T)
        # Upload the derivative of the weights to ElastiCache for use by `TrainerLambda`
        to_cache(endpoint=endpoint, obj=dw, name='layer'+str(layer)+'_dw_'+str(ID))
        
        # Debug statement
        assert(dw.shape == W.shape)

        # Calculate the derivatives of the bias
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True) #<-- Could be an issue here and may have to reshape in TrainerLambda when runnign backprop oin layer 1
        #db = 1 / m * np.sum(dZ)
        # Upload the erivative of the bis to ElastiCache for use by `TrainerLambda`
        to_cache(endpoint=endpoint, obj=db, name='layer'+str(layer)+'_db_'+str(ID))

        print("Completed Back Propogation for epoch {}, layer {}".format(str(epoch), str(layer)))

        if last == "True":
            # Update parameters with this Neuron's data
            parameters['epoch'] = epoch
            parameters['layer'] = layer - 1
            # Build the state payload
            payload = {}
            payload['parameter_key'] = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
            payload['state'] = 'backward'
            payload['epoch'] = epoch
            payload['layer'] = layer - 1

            # Crate an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "TrainerLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)

            # Debug Statement
            #print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))

            # Invoke TrainerLambda for next layer
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['TrainerLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking Trainer Lambda from NauronLambda."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
            #print(response)

        return

    else:
        sns_message = "General error processing NeuronLambda handler."
        publish_sns(sns_message)
        raise
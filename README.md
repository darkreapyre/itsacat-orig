# Serverless Neural Network for Image Classification - Demo

![alt text](https://github.com/darkreapyre/itsacat/blob/master/artifacts/images/Prediction_Architecture.png "Architecture")

## Pre-Requisites
1. This demo uses the [AWS CLI](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html). If the AWS CLI isn't installed,  follow [these](http://docs.aws.amazon.com/cli/latest/userguide/installing.html) instructions. The CLI [configuration](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) needs `PowerUserAccess` and `IAMFullAccess` [IAM policies](http://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html) associated. To verify that the AWS CLI is installed and up to date, run the following:
```console
    $ aws --version
```
2. Clone the repo.
```console
    $ git clone https://github.com/darkreapyre/itsacat
``` 

## Deployment
>**Note:** The demo has only been tested in the *us-west-2* Region.

2. Make any changes to the Neural Network configuraiton parameters file (`parameters.json`) before running the deployment.
2. To deploy the environment, change to the *deploy* directory. An easy to use deployment script has been created to automatically deploy the environment. Start the process by running `./deploy.sh`. You will be prompted for the following information:
```console
    Enter the AWS Region to use > <<AWS REGION>>
    Enter the S3 bucket to create > <<UNIQUE S3 BUCKET>>
    Enter the name of the Stack to deploy > <<UNIQUE CLOUDFOMRATION STACK NAME>>
    Enter the e-mail address to send training update > <<E-MAIL ADDRESS>>
```

3. The `deploy.sh` script creates the an *S3* Bucket; copies the necessary *CloudFormation* templates to the Bucket; creates the *Lambda* deployment package and uploads it to the Bucket and lastly, it creates the CloudFormation *Stack*. Once completed, the following message is displayed:

```console
    "Successfully created/updated stack - <<Stack Name>>"
```

>**Note:** During the deployment, and e-mail will be sent to the address specified in the `deploy.sh` script. Make sure to confirm the subscription to the SNS Topic.

## Integration with SageMaker Notebook Instance
Once the stack has been deployed, integrate [Amazon SageMaker](https://aws.amazon.com/sagemaker/) into the stack to start reviewing the Demo content by using the following steps:

1. Open the SageMaker [console](https://console.aws.amazon.com/sagemaker).
2. Create notebook instance.
3. Notebook instance settings.
    - Notebook instance name.
    - Notebook instance type -> ml.t2.medium.
    - IAM Role -> Create new role.
    - Specific S3 Bucket -> "UNIQUE BUCKET NAME" -> Create Role.
    - VPC -> "UNIQUE STACK NAME" .
    - Subnet -> select any of the subnets marked "Private".
    - Security group(s) -> HostSecurityGroup.
    - Create notebook instance.
3. You should see Status -> Pending.
4. Configure Service Access role.
    - IAM Console -> Role -> "AmazonSageMaker-ExecutionRole-...".
    - Policy name -> "AmazonSageMaker-ExecutionPolicy-..." -> Edit policy.
    - Visual editor tab -> Add additional permissions.
        - Service -> Choose a service -> ElastiCache.
        - Action -> Select and action -> All ElastiCache actions (elasticache:*) .
        - Review Policy.
        - Save changes.
    - The final Policy should look similar to this:
    ```json
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject",
                        "s3:GetObject",
                        "s3:ListBucket",
                        "s3:DeleteObject"
                    ],
                    "Resource": "arn:aws:s3:::*"
                },
                {
                    "Sid": "VisualEditor1",
                    "Effect": "Allow",
                    "Action": "elasticache:*",
                    "Resource": "*"
                }
            ]
        }
    ```
5. Return to the SageMaker console and confirm the Notebook instance is created. Under "Actions" -> Select "Open".
6. After the Jupyter Interface has opened, Configure the Python Libraries:
    - Select the "Conda" tab -> under "Conda environments" -> select "python3".
    - Under "Available packages" -> Click the "Search" -> enter "redis". The following two "redis" packages should be available. 
        - redis
        - redis-py
    - Select both packages and click the "->" button to install the packages.
    - Confirm to "Install" on the pop-up.
7. Clone the "itsacat" Code.
    - Under the "Files" tab -> Click "New" -> "Terminal".
    - Under the Shell run the following commands:
    ```shell
        $ cd SageMaker
        $ git clone https://github.com/darkreapyre/itsacat
        $ exit
    ```
    - Go back to the "Files" tab -> click "itsacat" -> click "artifacts" -> select `Introduction.ipynb`

## Jupyter Notebooks
### `Introduction.ipynb` Notebook
The **Introduction** provides an overview of the *Architecture*, *Why* and *How* the *Serverless Neural Network* is implemented.

### `Codebook.ipynb` Notebook
The **Codebook** provides an overview of the various *Python Libraries*, *Helper Functions* and the *Handler* functions that is integrated into each of the Lambda Functions. It also provides a mockup of a *2-Layer* implementation of the Neural Network using the code within the Notebook to get an understanding of the full training process will be executed.

## Training the Classifier
To train the full classification model on the *SNN* framework, simply upload the `datasets.h5` file found in the `datasets` directory to the `training_iput` folder that has already been created by the deployment process.
>**Note:** A pre-configured `parameters.json` file has already been created. To change the Neural Network configuration parameters, before running the training process, change this file and upload it to the `training_iput` folder of the S3 Bucket before uploading the data set.

Once the data file has been uploaded, an S3 Bucket Event will automatically trigger the training process. Should trigger process be successful, an automatic message will be sent to the e-mail address configured during deployment. The message should look as follows:
```text
    Training update!
    Cost after epoch 0 = 0.7012303687667679
```
In-depth insight to the training process can be viewed through the **CloudWatch** console.

If a message is not received after *5 minutes*, refer to the **Troubleshooting** section.

## Analyzing the Results
Once the training process has successfully completed, an e-mail will be sent to the address configured during the deployment. To analyze the results of the testing and to determine if the trained model is production-worthy, using the same *SageMaker* instance used for the *Codebook*, navigate to the `artifacts` directory and launch the `Analysis.ipynb` notebook.

Work through the various code cells to see:
1. Results fo the training.
2. How well the model performs against the **Test** dataset.
3. How well the model performs against new images.

>**Note:** Ensure to add the name of the S3 Bucket and AWS Region used during deployment to get the correct results files created during the training process.

## Troubleshooting
Since the framework launches a significant amount to Asynchronous Lambda functions without any pre-warming, the **CloudWatch** logs may show the following error:

```python
    list index out of range: IndexError
    Traceback (most recent call last):
    File "/var/task/trainer.py", line 484, in lambda_handler
    A = vectorizer(Outputs='a', Layer=layer-1)
    File "/var/task/trainer.py", line 235, in vectorizer
    key_list.append(result[0])
    IndexError: list index out of range
```

To address this, simply delete the data set from the S3 Bucket and re-upload it to re-launch the training process.

## Cleanup

1. Delete the CloudFormnation Stack.
    - Open the CloudFormation Service console.
    - Select the "bottom" stack in the **Stack Name** column.
    - Actions -> Delete Stack -> "Yes, Delete".
2. Delete DynamoDB Tables.
    - Open DynamoDB Service console.
    - Select "Tables" in the navigation panel.
    - Check *NeuronLambda* -> Delete table -> Delete.
    - Repeat the above process for the *TrainerLambda* table.
3. Delete the CloudWatch Logs.
    - Open the CloudWatch Service console.
    - Select "Logs" in the navigation panel.
    - Check */aws/lambda/LaunchLambda* -> Actions -> Delete log group -> Yes, Delete.
    - Repeat the above process for */aws/lambda/NeuronLambda*, */aws/lambda/S3TriggerLambda*, and */aws/lambda/S3TriggerLambda*.
4. Delete the S3 Bucket.
    - Open the S3 Service console.
    - Highlite the bucket -> Delete bucket.
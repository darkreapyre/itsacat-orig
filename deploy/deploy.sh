#!/bin/bash

echo -n "Enter the AWS Region to use >"
read region
echo -n "Enter the S3 bucket to create >"
read bucket
echo -n "Enter the name of the Stack to deploy >"
read stackname
echo -n "Enter the e-mail address to send training update >"
read email
export BUCKET=$bucket
export EMAIL=$email

echo ""
echo -n "Building main CloudFormation Template..."
rm -f main.yaml temp.yaml  
( echo "cat <<EOF >main.yaml";
  cat template.yaml;
  echo "EOF";
) >temp.yaml
. temp.yaml
rm -f temp.yaml

echo ""
echo -n "Building Lambda Deployment Package..."
rm -rf tmp
mkdir tmp
cd tmp
cp ../../lambda/src/*.py .
cp ../../lambda/deps.zip package.zip
zip -r package.zip *.py
mv package.zip ../lambda/train/
cd ..

echo ""
echo -n "Uploading Cloudformation Templates to S3..."
rm -rf tmp
mkdir tmp
zip tmp/templates.zip main.yaml infrastructure/*
aws s3 mb "s3://${bucket}"
aws s3 cp tmp/templates.zip "s3://${bucket}/deploy/" --region "${region}"
aws s3 cp main.yaml "s3://${bucket}/deploy/" --region "${region}"
aws s3 cp --recursive infrastructure/ "s3://${bucket}/deploy/infrastructure" --region "${region}"
aws s3 cp --recursive lambda/ "s3://${bucket}/deploy/lambda" --region "${region}"
aws s3 cp parameters.json "s3://${bucket}/training_input/" --region "${region}"
rm lambda/train/package.zip
rm -rf tmp

echo -m "Launching CloudFormation Stack"
aws cloudformation deploy --stack-name $stackname --template-file main.yaml --capabilities CAPABILITY_NAMED_IAM --parameter-overrides TemplateBucket=$bucket TopicEmail=$email --region $region
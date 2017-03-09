from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import time
import sys
import base64
import oauth2client.service_account
import googleapiclient.discovery
import google.cloud.storage


parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True, help="Google Cloud Storage bucket to upload to")
parser.add_argument("--model_name", required=True, help="name of Google Cloud Machine Learning model to create or update")
parser.add_argument("--model_dir", required=True, help="path to directory containing exported model")
parser.add_argument("--runtime_version", default="0.12", help="tensorflow version to use for the model")
parser.add_argument("--credentials", help="JSON credentials for a Google Cloud Platform service account")
parser.add_argument("--project", help="Google Cloud Project to use to override project detection")
a = parser.parse_args()

scopes = ["https://www.googleapis.com/auth/cloud-platform"]
if a.credentials is None:
    credentials = oauth2client.client.GoogleCredentials.get_application_default()
    storage = google.cloud.storage.Client()
    project_id = storage.project
    if a.project is not None:
        project_id = a.project
else:
    credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(a.credentials, scopes)
    with open(a.credentials) as f:
        project_id = json.loads(f.read())["project_id"]
    storage = google.cloud.storage.Client.from_service_account_json(a.credentials, project=project_id)

ml = googleapiclient.discovery.build("ml", "v1beta1", credentials=credentials)


def main():
    try:
        bucket = storage.get_bucket(a.bucket)
    except google.cloud.exceptions.NotFound as e:
        print("creating bucket %s" % a.bucket)
        bucket = storage.create_bucket(a.bucket)
    
    project_path = "projects/%s" % project_id
    model_path = "%s/models/%s" % (project_path, a.model_name)

    try:
        ml.projects().models().get(name=model_path).execute()
    except googleapiclient.errors.HttpError as e:
        if e.resp["status"] != "404":
            raise
        print("creating model %s" % a.model_name)
        ml.projects().models().create(parent=project_path, body=dict(name=a.model_name)).execute()

    version_number = 0
    resp = ml.projects().models().versions().list(parent=model_path).execute()
    for version in resp.get("versions", []):
        name = version["name"]
        number = int(name.split("/")[-1][1:])
        if number > version_number:
            version_number = number
    
    version_number += 1
    print("creating version v%d" % version_number)

    for filename in os.listdir(a.model_dir):
        if not filename.startswith("export.") and filename != "checkpoint":
            continue

        print("uploading", filename)
        filepath = os.path.join(a.model_dir, filename)
        blob = bucket.blob("%s-v%d/%s" % (a.model_name, version_number, filename))
        blob.upload_from_filename(filepath)

    version_path = "%s/versions/v%d" % (model_path, version_number)
    version = dict(
        name="v%d" % version_number,
        runtimeVersion=a.runtime_version,
        deploymentUri="gs://%s/%s-v%d/" % (a.bucket, a.model_name, version_number),
    )
    operation = ml.projects().models().versions().create(parent=model_path, body=version).execute()

    sys.stdout.write("waiting for creation to finish")
    while True:
        operation = ml.projects().operations().get(name=operation["name"]).execute()
        if "done" in operation and operation["done"]:
            break
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(10)
    print()

    print("setting version %d as default" % version_number)
    ml.projects().models().versions().setDefault(name=version_path, body=dict()).execute()


main()
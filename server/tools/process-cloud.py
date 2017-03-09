from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import base64
import oauth2client.service_account
import googleapiclient.discovery


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="name of Cloud Machine Learning model")
parser.add_argument("--input_file", required=True, help="input PNG image file")
parser.add_argument("--output_file", required=True, help="output PNG image file")
parser.add_argument("--credentials", required=True, help="JSON credentials for a Google Cloud Platform service account")
a = parser.parse_args()

scopes = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(a.credentials, scopes)
ml = googleapiclient.discovery.build("ml", "v1beta1", credentials=credentials)


def main():
    with open(a.credentials) as f:
        project_id = json.loads(f.read())["project_id"]

    with open(a.input_file) as f:
        input_data = f.read()

    input_instance = dict(input=base64.urlsafe_b64encode(input_data), key="0")
    input_instance = json.loads(json.dumps(input_instance))
    request = ml.projects().predict(name="projects/" + project_id + "/models/" + a.model_name, body={"instances": [input_instance]})
    response = request.execute()
    output_instance = json.loads(json.dumps(response["predictions"][0]))

    b64data = output_instance["output"].encode("ascii")
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data)

    with open(a.output_file, "w") as f:
        f.write(output_data)

main()
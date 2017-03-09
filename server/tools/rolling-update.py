from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import json


def main():
    output = subprocess.check_output("gcloud compute instance-groups managed list-instances pix2pix-manager --zone us-central1-c --format json", shell=True)
    instances = json.loads(output)
    for i, instance in enumerate(instances):
        name = instance["instance"].split("/")[-1]
        print("recreating %s (%d/%d)" % (name, i+1, len(instances)))
        subprocess.check_call("gcloud compute instance-groups managed recreate-instances pix2pix-manager --zone us-central1-c --instances " + name, shell=True)
        subprocess.check_call("gcloud compute instance-groups managed wait-until-stable pix2pix-manager --zone us-central1-c", shell=True)

main()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True, help="version to build")
parser.add_argument("--project", required=True, help="Google Cloud Project to use")
a = parser.parse_args()

def main():
    version_tag = "us.gcr.io/%s/pix2pix-server:%s" % (a.project, a.version)
    latest_tag = "us.gcr.io/%s/pix2pix-server:latest" % (a.project)

    subprocess.check_call("docker build --tag %s ." % version_tag, shell=True)
    subprocess.check_call("docker tag %s %s" % (version_tag, latest_tag), shell=True)
    for tag in [version_tag, latest_tag]:
        subprocess.check_call("gcloud docker -- push %s" % tag, shell=True)

main()

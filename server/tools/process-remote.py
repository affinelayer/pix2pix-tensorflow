from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from urllib.request import urlopen # python 3
except ImportError:
    from urllib2 import urlopen # python 2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, help="input PNG image file")
parser.add_argument("--url", required=True, help="url to use for processing")
parser.add_argument("--output_file", required=True, help="output PNG image file")
a = parser.parse_args()


def main():
    with open(a.input_file, "rb") as f:
        input_data = f.read()

    output_data = urlopen(a.url, data=input_data).read()

    with open(a.output_file, "wb") as f:
        f.write(output_data)

main()

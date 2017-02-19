from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import urlparse
import time
import argparse
import base64
import os
import json
import traceback

# https://github.com/Nakiami/MultithreadedSimpleHTTPServer/blob/master/MultithreadedSimpleHTTPServer.py
try:
    # Python 2
    from SocketServer import ThreadingMixIn
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    # Python 3
    from socketserver import ThreadingMixIn
    from http.server import HTTPServer, BaseHTTPRequestHandler

socket.setdefaulttimeout(30)

parser = argparse.ArgumentParser()
parser.add_argument("--local_models_dir", help="directory containing local models to serve (either this or --cloud_model_names must be specified)")
parser.add_argument("--cloud_model_names", help="comma separated list of cloud models to serve (either this or --local_models_dir must be specified)")
parser.add_argument("--addr", default="", help="address to listen on")
parser.add_argument("--port", default=8000, type=int, help="port to listen on")
parser.add_argument("--credentials", help="JSON credentials for a Google Cloud Platform service account, generate this at https://console.cloud.google.com/iam-admin/serviceaccounts/project (select \"Furnish a new private key\")")
parser.add_argument("--project", help="Google Cloud Project to use, only necessary if using default application credentials")
a = parser.parse_args()


models = {}
local = a.local_models_dir is not None
ml = None
project_id = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not os.path.exists("static"):
            self.send_response(404)
            return
            
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("static/index.html") as f:
                self.wfile.write(f.read())
            return

        filenames = [name for name in os.listdir("static") if not name.startswith(".")]
        path = self.path[1:]
        if path not in filenames:
            self.send_response(404)
            return

        self.send_response(200)
        if path.endswith(".png"):
            self.send_header("Content-Type", "image/png")
        elif path.endswith(".jpg"):
            self.send_header("Content-Type", "image/jpeg")
        else:
            self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        with open("static/" + path) as f:
            self.wfile.write(f.read())


    def do_OPTIONS(self):
        self.send_response(200)
        if "origin" in self.headers:
            self.send_header("access-control-allow-origin", "*")

        allow_headers = self.headers.get("access-control-request-headers", "*")
        self.send_header("access-control-allow-headers", allow_headers)
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.send_header("access-control-max-age", "3600")
        self.end_headers()


    def do_POST(self): 
        start = time.time()

        status = 200
        headers = {}
        body = ""

        try:
            name = self.path[1:]
            if name not in models:
                raise Exception("invalid model")

            content_len = int(self.headers.getheader("content-length", 0))
            if content_len > 1 * 1024 * 1024:
                raise Exception("post body too large")
            input_data = self.rfile.read(content_len)
            input_b64data = base64.urlsafe_b64encode(input_data)

            if local:
                m = models[name]
                output_b64data = m["sess"].run(m["output"], feed_dict={m["input"]: [input_b64data]})[0]
            else:
                input_instance = dict(input=input_b64data, key="0")
                request = ml.projects().predict(name="projects/" + project_id + "/models/" + name, body={"instances": [input_instance]})
                response = request.execute()
                output_instance = response["predictions"][0]
                output_b64data = output_instance["output"].encode("ascii")

            # add any missing padding
            output_b64data += "=" * (-len(output_b64data) % 4)
            output_data = base64.urlsafe_b64decode(output_b64data)
            headers["content-type"] = "image/png"
            body = output_data
        except Exception as e:
            print("exception", traceback.format_exc())
            status = 500
            body = "server error"

        self.send_response(status)
        if "origin" in self.headers:
            self.send_header("access-control-allow-origin", "*")
        for key, value in headers.iteritems():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

        print("finished in %0.1fs" % (time.time() - start))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def main():
    if a.local_models_dir is not None:
        import tensorflow as tf
        for name in os.listdir(a.local_models_dir):
            if name.startswith("."):
                continue

            print("loading model", name)

            with tf.Graph().as_default() as graph:
                sess = tf.Session(graph=graph)
                saver = tf.train.import_meta_graph(os.path.join(a.local_models_dir, name, "export.meta"))

                saver.restore(sess, os.path.join(a.local_models_dir, name, "export"))
                input_vars = json.loads(tf.get_collection("inputs")[0])
                output_vars = json.loads(tf.get_collection("outputs")[0])
                input = graph.get_tensor_by_name(input_vars["input"])
                output = graph.get_tensor_by_name(output_vars["output"])

                models[name] = dict(
                    sess=sess,
                    input=input,
                    output=output,
                )
    elif a.cloud_model_names is not None:
        import oauth2client.service_account
        import googleapiclient.discovery
        for name in a.cloud_model_names.split(","):
            models[name] = None

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        global project_id
        if a.credentials is None:
            credentials = oauth2client.client.GoogleCredentials.get_application_default()
            # use this only to detect the project
            import google.cloud.storage
            storage = google.cloud.storage.Client()
            project_id = storage.project
            if a.project is not None:
                project_id = a.project
        else:
            credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(a.credentials, scopes)
            with open(a.credentials) as f:
                project_id = json.loads(f.read())["project_id"]

        global ml
        ml = googleapiclient.discovery.build("ml", "v1beta1", credentials=credentials)
    else:
        raise Exception("must specify --local_models_dir or --cloud_model_names")

    print("listening on %s:%s" % (a.addr, a.port))
    ThreadedHTTPServer((a.addr, a.port), Handler).serve_forever()

main()

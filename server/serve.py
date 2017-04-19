from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import time
import argparse
import base64
import os
import json
import traceback
import threading
import multiprocessing
import random


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
parser.add_argument("--origin", help="allowed origin")
parser.add_argument("--addr", default="", help="address to listen on")
parser.add_argument("--port", default=8000, type=int, help="port to listen on")
parser.add_argument("--wait", default=0, type=int, help="time to wait for each request")
parser.add_argument("--credentials", help="JSON credentials for a Google Cloud Platform service account, generate this at https://console.cloud.google.com/iam-admin/serviceaccounts/project (select \"Furnish a new private key\")")
parser.add_argument("--project", help="Google Cloud Project to use, only necessary if using default application credentials")
a = parser.parse_args()

jobs = threading.Semaphore(multiprocessing.cpu_count() * 4)
models = {}
ml = None
project_id = None
build_cloud_client = None


class RateCounter(object):
    def __init__(self, window_us, granularity=1000):
        self.granularity = granularity
        self.width_us = int(window_us) // self.granularity
        self.buckets = [0] * self.granularity
        self.updated = 0
        self.lock = threading.RLock()

    def incr(self, amt=1):
        with self.lock:
            now_us = int(time.time() * 1e6)
            now = now_us // self.width_us

            if now > self.updated:
                # need to clear any buckets between the update time and now
                if now - self.updated > self.granularity:
                    self.buckets = [0] * self.granularity
                    self.updated = now
                else:
                    while self.updated <= now:
                        self.updated += 1
                        self.buckets[self.updated % self.granularity] = 0

            self.buckets[now % self.granularity] += amt

    def value(self):
        with self.lock:
            # update any expired buckets
            self.incr(amt=0)
            return sum(self.buckets)


successes = RateCounter(1 * 60 * 1e6)
failures = RateCounter(1 * 60 * 1e6)
cloud_requests = RateCounter(5 * 60 * 1e6)
cloud_accepts = RateCounter(5 * 60 * 1e6)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write("OK")
            return

        if not os.path.exists("static"):
            self.send_response(404)
            return

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("static/index.html", "rb") as f:
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
        with open("static/" + path, "rb") as f:
            self.wfile.write(f.read())


    def do_OPTIONS(self):
        self.send_response(200)
        if "origin" in self.headers:
            if a.origin is not None and self.headers["origin"] != a.origin:
                print("invalid origin %s" % self.headers["origin"])
                self.send_response(400)
                return
            self.send_header("access-control-allow-origin", self.headers["origin"])

        allow_headers = self.headers.get("access-control-request-headers", "*")
        self.send_header("access-control-allow-headers", allow_headers)
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.send_header("access-control-max-age", "3600")
        self.end_headers()


    def do_POST(self):
        start = time.time()

        status = 200
        headers = {}
        if "origin" in self.headers:
            headers = {"access-control-allow-origin": self.headers["origin"]}
        body = ""

        try:
            name = self.path[1:]

            if name not in models:
                raise Exception("invalid model")

            variants = models[name]  # "cloud" and "local" are the two possible variants

            content_len = int(self.headers.get("content-length", "0"))
            if content_len > 1 * 1024 * 1024:
                raise Exception("post body too large")
            input_data = self.rfile.read(content_len)
            input_b64data = base64.urlsafe_b64encode(input_data)

            time.sleep(a.wait)

            cloud_reject_prob = max(0, (cloud_requests.value() - 1.1 * cloud_accepts.value()) / (cloud_requests.value() + 1))
            # print("requests=%d accepts=%d cloud_reject_prob=%f" % (cloud_requests.value(), cloud_accepts.value(), cloud_reject_prob))

            output_b64data = None
            if "cloud" in variants and random.random() > cloud_reject_prob:
                input_instance = dict(input=input_b64data, key="0")
                # the client does not seem to be threadsafe, so make one for each request
                # also the cache is broken by oauth2client 4.0.0, so use a memory cache
                request = build_cloud_client().projects().predict(name="projects/" + project_id + "/models/" + name, body={"instances": [input_instance]})
                try:
                    cloud_requests.incr()
                    response = request.execute()
                    output_instance = response["predictions"][0]
                    output_b64data = output_instance["output"].encode("ascii")
                    cloud_accepts.incr()
                except Exception as e:
                    print("exception while running cloud model", traceback.format_exc())
                    print("falling back to local")

            if output_b64data is None and "local" in variants and jobs.acquire(blocking=False):
                m = variants["local"]
                try:
                    output_b64data = m["sess"].run(m["output"], feed_dict={m["input"]: [input_b64data]})[0]
                finally:
                    jobs.release()

            if output_b64data is None:
                raise Exception("too many requests")

            # add any missing padding
            output_b64data += b"=" * (-len(output_b64data) % 4)
            output_data = base64.urlsafe_b64decode(output_b64data)
            if output_data.startswith(b"\x89PNG"):
                headers["content-type"] = "image/png"
            else:
                headers["content-type"] = "image/jpeg"
            body = output_data
            successes.incr()
        except Exception as e:
            failures.incr()
            print("exception", traceback.format_exc())
            status = 500
            body = "server error"

        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

        print("finished in %0.1fs successes=%d failures=%d" % (time.time() - start, successes.value(), failures.value()))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def main():
    if a.local_models_dir is None and a.cloud_model_names is None:
        raise Exception("must specify --local_models_dir or --cloud_model_names")

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

                if name not in models:
                    models[name] = {}

                models[name]["local"] = dict(
                    sess=sess,
                    input=input,
                    output=output,
                )

    if a.cloud_model_names is not None:
        import oauth2client.service_account
        import googleapiclient.discovery
        import googleapiclient.discovery_cache.base
        import httplib2

        for name in a.cloud_model_names.split(","):
            if name not in models:
                models[name] = {}
            models[name]["cloud"] = None

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
            with open(a.credentials, "r") as f:
                project_id = json.loads(f.read())["project_id"]

        # due to what appears to be a bug, we cannot get the discovery document when specifying an http client
        # so grab it first, then the second build should use the cache
        class Cache(googleapiclient.discovery_cache.base.Cache):
            def __init__(self):
                self.cache = {}

            def get(self, url):
                return self.cache.get(url)

            def set(self, url, content):
                self.cache[url] = content

        cache = Cache()
        googleapiclient.discovery.build("ml", "v1beta1", credentials=credentials, cache=cache)
        global build_cloud_client
        build_cloud_client = lambda: googleapiclient.discovery.build("ml", "v1beta1", http=credentials.authorize(httplib2.Http(timeout=10)), cache=cache)

    print("listening on %s:%s" % (a.addr, a.port))
    ThreadedHTTPServer((a.addr, a.port), Handler).serve_forever()

main()

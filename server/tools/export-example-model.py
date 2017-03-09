from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True, help="directory to put exported model in")
a = parser.parse_args()


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    input = tf.placeholder(tf.string, shape=[1])
    key = tf.placeholder(tf.string, shape=[1])

    in_data = tf.decode_base64(input[0])
    img = tf.image.decode_png(in_data)
    img = tf.image.rgb_to_grayscale(img)
    out_data = tf.image.encode_png(img)
    output = tf.convert_to_tensor([tf.encode_base64(out_data)])

    variable_to_allow_model_saving = tf.Variable(1, dtype=tf.float32)

    inputs = {
        "key": key.name,
        "input": input.name
    }
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {
        "key":  tf.identity(key).name,
        "output": output.name,
    }
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
        saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)
    
    print("exported example model to %s" % a.output_dir)

main()

import os
import argparse
from pprint import pprint

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", default=None, help="output_dir from --mode export run")
    parser.add_argument("--frozen", default=None, help="frozen graph pb file")
    parser.add_argument("--tflite", default=None, help="tflite file")
    parser.add_argument("--input", default='Untitled.png', help="input image")
    parser.add_argument("--output", default='out.png', help="output image")
    a = parser.parse_args()

    out_files = [ a.output ]
    im_files = [ a.input ]
    images_cv = [ cv2.resize(cv2.imread(f), (256, 256)) for f in im_files ]
    images = np.array(images_cv, dtype=np.float32)
    images = images / 255.0

    if a.tflite:
        interpreter = tf.lite.Interpreter(model_path=a.tflite)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

        interpreter.set_tensor(input_details[0]['index'], images)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[:,:,:,::-1]
        output = output * 255

        print("Writing " + out_files[0])
        cv2.imwrite(out_files[0], output[0]);

    if a.frozen:
        with tf.Session() as sess:
            with gfile.FastGFile(a.frozen, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

            input = tf.get_default_graph().get_tensor_by_name("TFLiteInput:0")
            output = tf.get_default_graph().get_tensor_by_name("TFLiteOutput:0")
            output = output[:,:,:,::-1]
            output = output * 255

            print("Writing " + out_files[0])
            cv2.imwrite(out_files[0], output.eval({'TFLiteInput:0': images})[0]);

    if a.export:
        with tf.Session() as sess:
            with gfile.FastGFile(os.path.join(a.export, 'export.meta'), 'rb') as f:
                meta_graph_def = tf.MetaGraphDef()
                meta_graph_def.ParseFromString(f.read())
                tf.train.import_meta_graph(meta_graph_def)
                checkpoint = tf.train.latest_checkpoint(a.export)
                restore_saver = tf.train.Saver()
                restore_saver.restore(sess, checkpoint)

            input = tf.get_default_graph().get_tensor_by_name("TFLiteInput:0")
            output = tf.get_default_graph().get_tensor_by_name("TFLiteOutput:0")
            output = output[:,:,:,::-1]
            output = output * 255

            print("Writing " + out_files[0])
            cv2.imwrite(out_files[0], output.eval({'TFLiteInput:0': images})[0]);

main()

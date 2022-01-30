from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import rospy
import datetime
import numpy as np
import os
import time

THRESHOLD = 0.5
MODEL_PATH = "models/ssd_mobilenet_v2/graph_optimized.pb"


class TLClassifier(object):
    def __init__(self, config):
        self.image_size = (config['camera_info']['image_height'],
                           config['camera_info']['image_width'])
        self.threshold = THRESHOLD
        self.total_time = 0
        self.num_detections = 0
        self.load_model(MODEL_PATH)
        self.class_map = {
            1: TrafficLight.GREEN,
            2: TrafficLight.YELLOW,
            3: TrafficLight.RED
        }

        # Initialize the node's logger
        self.logger = Logger()

        self.warmup_model()

    def load_model(self, model_path):
        base_folder = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(base_folder, model_path)

        self.logger.warn('Loading model: %s', model_path)

        graph = tf.Graph()

        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=graph)

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.boxes_tensor = graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = graph.get_tensor_by_name('detection_classes:0')
        self.detections_tensor = graph.get_tensor_by_name('num_detections:0')

    def warmup_model(self):
        image = np.zeros((self.image_size[0], self.image_size[1], 3),
                         dtype=np.uint8)
        _, elapsed_time = self.detect_light(image)
        self.logger.warn('Tensorflow warmup completed (Time elapsed: %.3f ms)',
                         elapsed_time)

    def get_detected_class(self, detection_scores, detection_classes):
        if detection_scores[0] >= self.threshold:
            return self.class_map.get(detection_classes[0],
                                      TrafficLight.UNKNOWN)
        return TrafficLight.UNKNOWN

    def detect_light(self, image):
        input_image = np.expand_dims(image, axis=0)

        s_time = time.time()
        ops = [self.detections_tensor, self.boxes_tensor, self.scores_tensor,
               self.classes_tensor]
        _, _, detection_scores, detection_classes = self.sess.run(
            ops, feed_dict={self.image_tensor: input_image})
        e_time = time.time() - s_time

        detection_scores = detection_scores[0]
        detection_classes = detection_classes[0].astype(np.uint8)

        return self.get_detected_class(detection_scores,
                                       detection_classes), e_time * 1000.0

    def get_classification(self, image_rgb):
        """Determines the color of the traffic light in the image
        Args:
            image_rgb (cv::Mat): image (RGB) containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        detected_light, elapsed_time = self.detect_light(image_rgb)
        self.total_time += elapsed_time
        self.num_detections += 1

        if self.num_detections % 50 == 0:
            self.logger.warn('Detections: %s, Avg Detection Time: %.3f ms',
                             self.num_detections,
                             self.total_time / self.num_detections)

        return detected_light

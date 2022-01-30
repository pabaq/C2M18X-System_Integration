#!/usr/bin/env python
import math

import rospy
import tf
import cv2
import yaml
import numpy as np

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier

from utilities import *

USE_LIGHT_STATE = False
STATE_COUNT_THRESHOLD = 3
SYNC_QUEUE_SIZE = 20

'''
/vehicle/traffic_lights provides you with the location of the traffic light in 
3D map space and helps you acquire an accurate ground truth data source for the 
traffic light classifier by sending the current color state of all traffic 
lights in the simulator. When testing on the vehicle, the color state will not 
be available. You'll need to rely on the position of the light and the camera 
image to predict it. 
'''


class TLDetector(object):
    def __init__(self):
        # Initialize the /tl_detector node
        rospy.init_node('tl_detector')
        # Subscribe to the required topics, which are
        # - the reference waypoints to be followed by the ego vehicle
        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)
        # - the current pose of the ego vehicle
        # rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        synced_subscribers = [Subscriber('/current_pose', PoseStamped)]
        # - the current twist of the ego vehicle
        rospy.Subscriber('/current_velocity', TwistStamped,
                         self.current_velocity_cb)
        # - the location of all traffic ligths on the map
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                         self.traffic_lights_cb)
        # - the images taken from the car's camera
        # rospy.Subscriber('/image_color', Image, self.image_color_cb)

        # Initialize
        # - the /traffic_waypoint publisher, which will provide the waypoint
        #   index of the nearest red traffic light in front of the ego vehicle
        self.traffic_light_pub = rospy.Publisher('/traffic_waypoint', Int32,
                                                 queue_size=1)
        # - the WaypointTree instance of the reference waypoints
        self.tree = None  # type: WaypointTree | None
        # - the ego vehicle's current pose
        self.current_pose = None  # type: PoseStamped | None
        # - the ego vehicle's current twist
        self.current_twist = None  # type: TwistStamped | None
        # - the list of traffic lights on the map
        self.traffic_lights = []  # type: list[TrafficLight]
        # - the current camera image
        self.camera_image = None

        # Initialize other required attributes
        # Load the traffic light's stop line positions and the camera info
        self.config = yaml.load(rospy.get_param("/traffic_light_config"))
        self.stoplines = self.config["stop_line_positions"]  # type: list[list]
        self.current_light_state = TrafficLight.UNKNOWN
        self.current_stopline_idx = -1
        self.light_state_count = 0
        # self.previous_light_location = None
        self.has_image = False
        # Initialize the node's logger
        self.logger = Logger()

        # Check if we force the usage of the simulator light state, not
        # available when on site
        if USE_LIGHT_STATE and not self.config['is_site']:
            self.logger.warn('Classifier disabled, using simulator light state')
            # Note that we do not subscribe to the camera image
        else:
            # Setup the classifier
            self.light_classifier = TLClassifier(self.config)
            # When the classifier is enabled then the camera image needs to be
            # in sync with the current_pose
            synced_subscribers.append(Subscriber('/image_color', Image))
            # TODO Find out, this should greatly improve performance:
            # Subscriber('/image_color', Image, queue_size=1, buff_size=???)
            # buff_size should be greater than queue_size * avg_msg_byte_size

        # TODO: How does this work?
        self.synced_sub = ApproximateTimeSynchronizer(synced_subscribers,
                                                      SYNC_QUEUE_SIZE, 0.1)
        self.synced_sub.registerCallback(self.synced_data_cb)

        # TODO: Implement classifier later
        self.bridge = CvBridge()

        rospy.spin()

    # def current_pose_cb(self, current_pose):
    #     """ Store the current pose of the ego vehicle.
    #
    #     Args:
    #         current_pose (PoseStamped)
    #     """
    #     self.current_pose = current_pose

    def current_velocity_cb(self, current_twist):
        """ Store the current velocity of the ego vehicle.

        Args:
            current_twist (TwistStamped)
        """
        self.current_twist = current_twist

    def base_waypoints_cb(self, reference_waypoints):
        """ Initialize the reference waypoints' WaypointTree.

        Args:
            reference_waypoints (Lane)
        """

        # Since the /waypoint_loader node keeps publishing the same reference
        # waypoints all the time, they need to be stored only once.
        if self.tree is None:
            self.tree = WaypointTree(reference_waypoints)

    def traffic_lights_cb(self, traffic_ligths_array):
        """ Store the traffic light array.

        Args:
            traffic_ligths_array (TrafficLightArray)
        """
        self.traffic_lights = traffic_ligths_array.lights

    def synced_data_cb(self, current_pose, image=None):
        """ Process incoming camera image and publish stop line waypoint index.

        Identifies red lights in the incoming camera image and publishes the
        index of the waypoint closest to the red light's stop line to
        /traffic_waypoint

        Args:
            current_pose (PoseStamped):
                ego vehicle's current pose
            image (Image):
                image from car-mounted camera
        """
        self.logger.reset()
        self.current_pose = current_pose
        self.camera_image = image
        stopline_idx, light_state = self.detect_next_traffic_light()

        if self.current_light_state != light_state:
            self.light_state_count = 0
            self.current_light_state = light_state
        elif self.light_state_count >= STATE_COUNT_THRESHOLD:
            if light_state == TrafficLight.RED:
                self.current_stopline_idx = stopline_idx
            else:
                self.current_stopline_idx = UNKNOWN
            self.traffic_light_pub.publish(Int32(self.current_stopline_idx))
        else:
            self.traffic_light_pub.publish(Int32(self.current_stopline_idx))

        # TODO: Old
        # if self.current_stopline_idx != stopline_idx:
        #     self.current_stopline_idx = stopline_idx
        #     if stopline_idx != UNKNOWN:
        #         self.logger.warn("Detected traffic light.")
        #
        # # If the current light state changed
        # if self.current_light_state != light_state:
        #     self.logger.warn("The light's color changed to %s",
        #                      LIGHT_STATES[light_state])
        #     self.current_light_state = light_state  # Store the new light state
        # # If the current light state did not change
        # else:
        #     self.logger.info("Light color: %s", LIGHT_STATES[light_state])
        #
        # # Only stop lines of red lights will be handled
        # if light_state == TrafficLight.RED:
        #     self.traffic_light_pub.publish(Int32(stopline_idx))
        # else:
        #     self.traffic_light_pub.publish(Int32(UNKNOWN))

        self.light_state_count += 1

    def detect_next_traffic_light(self):
        """ Determine the closest visible traffic light and its color state.

        Returns:
            tuple of the stop line index and the traffic light color id. If no
             traffic light was detected, returns (UNKNOWN, TrafficLight.UNKNOWN)
        """

        closest_light = None  # type: TrafficLight | None
        stopline_idx = None  # type: int | None

        # Begin the search if the required data is initialized
        # TODO: Neue Bedingung
        if self.tree:
            num_lookahead = NUM_LOOKAHEAD_LIGHT
            # Index of closest reference waypoint to ego vehicle's current pose
            ego_idx = self.tree.get_closest_idx_from(self.current_pose)
            # Find closest upcoming traffic light within the lookahead distance
            for light, stopline_xy in zip(self.traffic_lights, self.stoplines):
                # Index of closest reference waypoint to current traffic light
                light_idx = self.tree.get_closest_idx_from(light.pose)
                # Number of waypoints beetween traffic light and ego vehicle
                num_waypoints = light_idx - ego_idx
                # If it is the closest traffic light ahead of the ego vehicle
                if 0 <= num_waypoints < num_lookahead:
                    # Store traffic light and its stopline as the closest ones
                    closest_light = light
                    stopline_idx = self.tree.get_closest_idx_from(stopline_xy)
                    # Update the lookahead distance
                    num_lookahead = num_waypoints

        # If a traffic light was found within the lookahead distance
        if closest_light:
            # Predict the traffic light's state
            light_state = self.get_light_state(closest_light, self.camera_image)
            # Return the traffic light's reference waypoint index and its state
            return stopline_idx, light_state

        # If no traffic light was found return dummy values
        return UNKNOWN, TrafficLight.UNKNOWN

    def get_light_state(self, traffic_light, image=None):
        """ Determines the current color of the traffic light

        Args:
            traffic_light (TrafficLight): light to classify
            image (Image): current camera image

        Returns:
            traffic light color ID (specified in styx_msgs/TrafficLight)

        """

        light_state = TrafficLight.UNKNOWN

        # If no classifier is available uses the light state
        if self.light_classifier is None:
            light_state = traffic_light.state
        elif image is not None:
            image_rgb = self.bridge.imgmsg_to_cv2(image, "rgb8")
            light_state = self.light_classifier.get_classification(image_rgb)

        return light_state

        # TODO: Implement classification Later
        # if not self.has_image:
        #     self.previous_light_location = None
        #     return False
        # Get classification
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # return self.light_classifier.get_classification(cv_image)

        # return traffic_light.state


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

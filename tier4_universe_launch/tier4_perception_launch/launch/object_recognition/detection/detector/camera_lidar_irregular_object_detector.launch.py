# Copyright 2025 Tier IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import launch
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import LoadComposableNodes
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
import yaml


class SmallUnknownPipeline:
    def __init__(self, context):
        self.camera_ids = LaunchConfiguration("fusion_camera_ids").perform(context)
        # convert string to list
        self.camera_ids = yaml.load(self.camera_ids, Loader=yaml.FullLoader)

        self.additional_params = {
            "rois_number": len(self.camera_ids),
        }

        for index, camera_id in enumerate(self.camera_ids):
            self.additional_params[f"input/rois{index}"] = (
                f"/perception/object_recognition/detection/rois{camera_id}"
            )
            self.additional_params[f"input/camera_info{index}"] = (
                f"/sensing/camera/camera{camera_id}/camera_info"
            )
            self.additional_params[f"input/image{index}"] = (
                f"/sensing/camera/camera{camera_id}/image_raw"
            )

        # Filter configurations used in fusion_camera_ids
        # TODO: It might be better to pass full configuration lists and let the node handle the filtering
        # (In which case the below code can be removed and the param file can directly be passed to the node)
        sync_param_path = PathJoinSubstitution(
            [
                FindPackageShare("autoware_launch_config"),
                "config/perception/object_recognition/detection/image_projection_based_fusion/fusion_common.param.yaml",
            ]
        ).perform(context)
        with open(sync_param_path, "r") as f:
            self.roi_pointcloud_fusion_sync_param = yaml.safe_load(f)["/**"]["ros__parameters"]

        rois_timestamp_offsets = []
        rois_timestamp_noise_window = []
        approximate_camera_projection = []
        point_project_to_unrectified_image = []

        for index, camera_id in enumerate(self.camera_ids):
            rois_timestamp_offsets.append(
                self.roi_pointcloud_fusion_sync_param["rois_timestamp_offsets"][camera_id]
            )
            rois_timestamp_noise_window.append(
                self.roi_pointcloud_fusion_sync_param["matching_strategy"][
                    "rois_timestamp_noise_window"
                ][camera_id]
            )
            approximate_camera_projection.append(
                self.roi_pointcloud_fusion_sync_param["approximate_camera_projection"][camera_id]
            )
            point_project_to_unrectified_image.append(
                self.roi_pointcloud_fusion_sync_param["point_project_to_unrectified_image"][
                    camera_id
                ]
            )

        self.roi_pointcloud_fusion_sync_param["rois_timestamp_offsets"] = rois_timestamp_offsets
        self.roi_pointcloud_fusion_sync_param["approximate_camera_projection"] = (
            approximate_camera_projection
        )
        self.roi_pointcloud_fusion_sync_param["matching_strategy"][
            "rois_timestamp_noise_window"
        ] = rois_timestamp_noise_window
        self.roi_pointcloud_fusion_sync_param["approximate_camera_projection"] = (
            approximate_camera_projection
        )
        self.roi_pointcloud_fusion_sync_param["point_project_to_unrectified_image"] = (
            point_project_to_unrectified_image
        )

    def get_agnocast_env(self):
        agnocast_env = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("autoware_agnocast_wrapper"),
                        "launch",
                        "agnocast_env.launch.py",
                    ]
                ),
            ),
        )
        return agnocast_env

    def create_irregular_object_pipeline(self, input_topic, concat_info_topic, output_topic):
        components = []
        # create cropbox filter
        components.append(
            ComposableNode(
                package="autoware_pointcloud_preprocessor",
                plugin="autoware::pointcloud_preprocessor::CropBoxFilterComponent",
                name="crop_box_filter",
                remappings=[
                    ("input", input_topic),
                    ("input/concatenation_info", concat_info_topic),
                    ("output", "cropped_range/pointcloud"),
                ],
                parameters=[
                    {
                        "input_frame": LaunchConfiguration("base_frame"),
                        "output_frame": LaunchConfiguration("base_frame"),
                    },
                    ParameterFile(
                        param_file=PathJoinSubstitution(
                            [
                                FindPackageShare("autoware_launch_config"),
                                "config/perception/object_recognition/detection/irregular_object_detection/crop_box_filter.param.yaml",
                            ],
                        ),
                    ),
                ],
                extra_arguments=[
                    {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                ],
            )
        )

        # create ground_segmentation
        components.append(
            ComposableNode(
                package="autoware_ground_segmentation",
                plugin="autoware::ground_segmentation::ScanGroundFilterComponent",
                name="ground_filter",
                remappings=[
                    ("input", "cropped_range/pointcloud"),
                    ("output", "obstacle_segmentation/pointcloud"),
                ],
                parameters=[
                    ParameterFile(
                        param_file=PathJoinSubstitution(
                            [
                                FindPackageShare("autoware_launch_config"),
                                "config/perception/object_recognition/detection/irregular_object_detection/ground_segmentation.param.yaml",
                            ],
                        ),
                    ),
                ],
                extra_arguments=[
                    {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                ],
            )
        )
        return components

    def create_roi_pointcloud_fusion_node(self, input_topic, output_topic):
        node = Node(
            package="autoware_image_projection_based_fusion",
            executable="roi_pointcloud_fusion_node",
            name="roi_pointcloud_fusion",
            remappings=[
                ("input", input_topic),
                ("output", output_topic),
            ],
            parameters=[
                self.roi_pointcloud_fusion_sync_param,
                # ParameterFile(
                #     param_file=PathJoinSubstitution(
                #         [
                #             FindPackageShare("autoware_launch_config"),
                #             "config/perception/object_recognition/detection/image_projection_based_fusion/fusion_common.param.yaml",
                #         ]
                #     ),
                # ),
                ParameterFile(
                    param_file=PathJoinSubstitution(
                        [
                            FindPackageShare("autoware_launch_config"),
                            "config/perception/object_recognition/detection/irregular_object_detection/roi_pointcloud_fusion.param.yaml",
                        ]
                    ),
                ),
                self.additional_params,
            ],
            additional_env={"LD_PRELOAD": LaunchConfiguration("ld_preload_value")},
        )
        return node


def launch_setup(context, *args, **kwargs):
    obstacle_pointcloud_topic = "obstacle_segmentation/pointcloud"
    pipeline = SmallUnknownPipeline(context)
    agnocast_env = pipeline.get_agnocast_env()
    components = []
    components.extend(
        pipeline.create_irregular_object_pipeline(
            LaunchConfiguration("input/pointcloud"),
            LaunchConfiguration("input/concatenation_info"),
            obstacle_pointcloud_topic,
        )
    )
    loader = LoadComposableNodes(
        composable_node_descriptions=components,
        target_container=LaunchConfiguration("pointcloud_container_name"),
    )
    roi_pointcloud_fusion_node = pipeline.create_roi_pointcloud_fusion_node(
        obstacle_pointcloud_topic, LaunchConfiguration("output_topic")
    )
    return [agnocast_env, loader, roi_pointcloud_fusion_node]


def generate_launch_description():
    launch_arguments = []

    def add_launch_arg(name: str, default_value=None):
        launch_arguments.append(DeclareLaunchArgument(name, default_value=default_value))

    add_launch_arg("input/pointcloud", "/sensing/lidar/concatenated/pointcloud")
    add_launch_arg("input/concatenation_info", "/sensing/lidar/concatenated/pointcloud_info")
    add_launch_arg(
        "output_topic", "/perception/object_recognition/detection/irregular_object/clusters"
    )
    add_launch_arg("base_frame", "base_link")
    add_launch_arg("use_intra_process", "True")
    add_launch_arg("use_multithread", "True")
    add_launch_arg("fusion_camera_ids", "[3,5]")
    add_launch_arg("image_topic_name", "image_raw")
    add_launch_arg("pointcloud_container_name", "pointcloud_container")
    add_launch_arg("use_pointcloud_container", "True")

    return launch.LaunchDescription(
        [
            *launch_arguments,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("autoware_agnocast_wrapper"),
                            "launch",
                            "agnocast_env.launch.py",
                        ]
                    ),
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ],
    )

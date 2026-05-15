# Copyright 2020 Tier IV, Inc. All rights reserved.
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
from launch.actions import OpaqueFunction
from launch.actions import SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


class GroundSegmentationPipeline:
    def __init__(self, context):
        self.context = context
        self.vehicle_info = self.get_vehicle_info()

    def get_vehicle_info(self):
        # TODO(TIER IV): Use Parameter Substitution after we drop Galactic support
        # https://github.com/ros2/launch_ros/blob/master/launch_ros/launch_ros/substitutions/parameter.py
        gp = self.context.launch_configurations.get("ros_params", {})
        if not gp:
            gp = dict(self.context.launch_configurations.get("global_params", {}))
        p = {}
        p["vehicle_length"] = gp["front_overhang"] + gp["wheel_base"] + gp["rear_overhang"]
        p["vehicle_width"] = gp["wheel_tread"] + gp["left_overhang"] + gp["right_overhang"]
        p["min_longitudinal_offset"] = -gp["rear_overhang"]
        p["max_longitudinal_offset"] = gp["front_overhang"] + gp["wheel_base"]
        p["min_lateral_offset"] = -(gp["wheel_tread"] / 2.0 + gp["right_overhang"])
        p["max_lateral_offset"] = gp["wheel_tread"] / 2.0 + gp["left_overhang"]
        p["min_height_offset"] = 0.0
        p["max_height_offset"] = gp["vehicle_height"]
        return p

    def create_common_pipeline(self, input_topic, output_topic):
        max_z = self.vehicle_info["max_height_offset"] + LaunchConfiguration(
            "common_crop_box_filter_margin_max_z"
        ).perform(self.context)
        min_z = self.vehicle_info["min_height_offset"] + LaunchConfiguration(
            "common_crop_box_filter_margin_min_z"
        ).perform(self.context)

        components = []
        components.append(
            ComposableNode(
                package="autoware_pointcloud_preprocessor",
                plugin="autoware::pointcloud_preprocessor::CropBoxFilterComponent",
                name="crop_box_filter",
                remappings=[
                    ("input", input_topic),
                    ("output", "range_cropped/pointcloud"),
                ],
                parameters=[
                    ParameterFile(
                        param_file=PathJoinSubstitution(
                            [
                                FindPackageShare("autoware_launch_config"),
                                "config/perception/obstacle_segmentation/ground_segmentation",
                                "common_crop_box_filter.param.yaml",
                            ]
                        ),
                        allow_substs=True,
                    ),
                    {
                        "input_frame": "base_link",
                        "output_frame": "base_link",
                        "max_z": max_z,
                        "min_z": min_z,
                    },
                ],
                extra_arguments=[
                    {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                ],
            )
        )

        components.append(
            ComposableNode(
                package="autoware_ground_segmentation",
                plugin="autoware::ground_segmentation::ScanGroundFilterComponent",
                name="common_ground_filter",
                remappings=[
                    ("input", "range_cropped/pointcloud"),
                    ("output", output_topic),
                ],
                parameters=[
                    ParameterFile(
                        param_file=PathJoinSubstitution(
                            [
                                FindPackageShare("autoware_launch_config"),
                                "config/perception/obstacle_segmentation/ground_segmentation",
                                "common_ground_filter.param.yaml",
                            ]
                        ),
                        allow_substs=True,
                    ),
                    self.vehicle_info,
                    {
                        "input_frame": "base_link",
                        "output_frame": "base_link",
                    },
                ],
                extra_arguments=[
                    {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                ],
            )
        )
        return components

    @staticmethod
    def create_time_series_outlier_filter_components(input_topic, output_topic):
        components = []
        components.append(
            ComposableNode(
                package="autoware_occupancy_grid_map_outlier_filter",
                plugin="autoware::occupancy_grid_map_outlier_filter::OccupancyGridMapOutlierFilterComponent",
                name="occupancy_grid_based_outlier_filter",
                remappings=[
                    ("~/input/occupancy_grid_map", "/perception/occupancy_grid_map/map"),
                    ("~/input/pointcloud", input_topic),
                    ("~/output/pointcloud", output_topic),
                ],
                parameters=[
                    ParameterFile(
                        param_file=PathJoinSubstitution(
                            [
                                FindPackageShare("autoware_occupancy_grid_map_outlier_filter"),
                                "config/occupancy_grid_map_outlier_filter.param.yaml",
                            ]
                        ),
                        allow_substs=True,
                    )
                ],
                extra_arguments=[
                    {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                ],
            )
        )

        return components


def launch_setup(context, *args, **kwargs):
    pipeline = GroundSegmentationPipeline(context)

    components = []
    components.extend(
        pipeline.create_common_pipeline(
            input_topic="/sensing/lidar/concatenated/pointcloud",
            output_topic="/perception/obstacle_segmentation/single_frame/pointcloud",
        )
    )
    components.extend(
        pipeline.create_time_series_outlier_filter_components(
            input_topic="/perception/obstacle_segmentation/single_frame/pointcloud",
            output_topic="/perception/obstacle_segmentation/pointcloud",
        )
    )
    return [
        LoadComposableNodes(
            composable_node_descriptions=components,
            target_container="/pointcloud_container",
        )
    ]


def generate_launch_description():
    return launch.LaunchDescription(
        [
            DeclareLaunchArgument("use_intra_process", default_value="True"),
            SetLaunchConfiguration(
                "common_crop_box_filter_margin_max_z",
                "0.0",
            ),
            SetLaunchConfiguration(
                "common_crop_box_filter_margin_min_z",
                "-2.5",
            ),
            OpaqueFunction(function=launch_setup),
        ],
    )

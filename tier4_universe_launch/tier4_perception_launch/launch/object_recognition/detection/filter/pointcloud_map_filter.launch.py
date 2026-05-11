# Copyright 2022 TIER IV, Inc. All rights reserved.
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
from launch.actions import SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_arguments = []

    def add_launch_arg(name: str, default_value=None):
        launch_arguments.append(DeclareLaunchArgument(name, default_value=default_value))

    add_launch_arg("input_topic", "")
    add_launch_arg("output_topic", "")
    add_launch_arg("use_intra_process", "True")
    add_launch_arg("pointcloud_container_name", "pointcloud_container")
    add_launch_arg("use_pointcloud_map", "true")

    # for compare map pipeline
    down_sample_topic = (
        "/perception/obstacle_segmentation/pointcloud_map_filtered/downsampled/pointcloud"
    )

    return launch.LaunchDescription(
        [
            *launch_arguments,
            SetLaunchConfiguration("down_sample_voxel_size", "0.1"),
            # no-compare map pipeline
            LoadComposableNodes(
                composable_node_descriptions=[
                    ComposableNode(
                        package="autoware_pointcloud_preprocessor",
                        plugin="autoware::pointcloud_preprocessor::ApproximateDownsampleFilterComponent",
                        name="voxel_grid_downsample_filter",
                        remappings=[
                            ("input", LaunchConfiguration("input_topic")),
                            ("output", LaunchConfiguration("output_topic")),
                        ],
                        parameters=[
                            {
                                "voxel_size_x": LaunchConfiguration("down_sample_voxel_size"),
                                "voxel_size_y": LaunchConfiguration("down_sample_voxel_size"),
                                "voxel_size_z": LaunchConfiguration("down_sample_voxel_size"),
                            }
                        ],
                        extra_arguments=[
                            {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                        ],
                    ),
                ],
                target_container=LaunchConfiguration("pointcloud_container_name"),
                condition=UnlessCondition(LaunchConfiguration("use_pointcloud_map")),
            ),
            # compare map pipeline
            LoadComposableNodes(
                composable_node_descriptions=[
                    ComposableNode(
                        package="autoware_pointcloud_preprocessor",
                        plugin="autoware::pointcloud_preprocessor::VoxelGridDownsampleFilterComponent",
                        name="voxel_grid_downsample_filter",
                        remappings=[
                            ("input", LaunchConfiguration("input_topic")),
                            ("output", down_sample_topic),
                        ],
                        parameters=[
                            {
                                "voxel_size_x": LaunchConfiguration("down_sample_voxel_size"),
                                "voxel_size_y": LaunchConfiguration("down_sample_voxel_size"),
                                "voxel_size_z": LaunchConfiguration("down_sample_voxel_size"),
                            }
                        ],
                        extra_arguments=[
                            {"use_intra_process_comms": LaunchConfiguration("use_intra_process")}
                        ],
                    ),
                    ComposableNode(
                        package="autoware_compare_map_segmentation",
                        plugin="autoware::compare_map_segmentation::VoxelBasedCompareMapFilterComponent",
                        name="voxel_based_compare_map_filter",
                        remappings=[
                            ("input", down_sample_topic),
                            ("map", "/map/pointcloud_map"),
                            ("output", LaunchConfiguration("output_topic")),
                            ("map_loader_service", "/map/get_differential_pointcloud_map"),
                            ("kinematic_state", "/localization/kinematic_state"),
                        ],
                        parameters=[
                            ParameterFile(
                                param_file=PathJoinSubstitution(
                                    [
                                        FindPackageShare("autoware_launch_config"),
                                        "config/perception/object_recognition/detection/pointcloud_filter/pointcloud_map_filter.param.yaml",
                                    ]
                                ),
                                allow_substs=True,
                            ),
                            {
                                "input_frame": "map",
                            },
                        ],
                        extra_arguments=[
                            {"use_intra_process_comms": False},
                        ],
                    ),
                ],
                target_container=LaunchConfiguration("pointcloud_container_name"),
                condition=IfCondition(LaunchConfiguration("use_pointcloud_map")),
            ),
        ]
    )

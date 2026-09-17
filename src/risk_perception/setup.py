from setuptools import find_packages, setup
from glob import glob

package_name = 'risk_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    #data_files=[
     #   ('share/ament_index/resource_index/packages',
    #        ['resource/' + package_name]),
   #     ('share/' + package_name, ['package.xml']),
  # 	('share/' + package_name + '/launch', glob('launch/*.launch.py')),
 #   	('share/' + package_name + '/config', glob('config/*.yaml')),
#    ],
   data_files=[
        ('share/ament_index/resource_index/packages', ['resource/risk_perception']),
        ('share/risk_perception', ['package.xml']),
        ('share/risk_perception/launch', glob('launch/*.launch.py')),   # <- add
        ('share/risk_perception/config', glob('config/*.yaml')),        # <- add
        ('share/risk_perception/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User A',
    maintainer_email='user-a@example.com',
    description='ROS2 perception noes for scene understanding and risk-aware navigation.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	        "image_wrapper_test = risk_perception.image_wrapper_test_node:main",
        	"gdino_detector = risk_perception.gdino_detector_node:main",
        	"sam2_segmenter = risk_perception.sam2_segmenter_node:main",
        	"rgbd_projector = risk_perception.rgbd_projector_node:main",
        	"map_frame_projector = risk_perception.map_frame_projector_node:main",
        	"risk_costmap = risk_perception.risk_costmap_node:main",
        	"predictive_risk_costmap = risk_perception.predictive_risk_costmap_node:main",
        	"spatial_prior = risk_perception.spatial_prior_node:main",
        	"global_cam_bridge     = risk_perception.global_cam_bridge_node:main",
		"global_cam_calibrator = risk_perception.global_cam_calibrator_node:main",
		"global_cam_projector  = risk_perception.global_cam_projector_node:main",
		"global_cam_localizer  = risk_perception.global_cam_localizer_node:main",
		"global_cam_survey     = risk_perception.global_cam_survey_node:main",
		"global_cam_map_align  = risk_perception.global_cam_map_align_node:main",
		"global_cam_tag_monitor = risk_perception.global_cam_tag_monitor_node:main",
		"global_cam_align_check = risk_perception.global_cam_align_check_node:main",
		"global_cam_initialpose = risk_perception.global_cam_initialpose_node:main",
		"object_tracker        = risk_perception.object_tracker_node:main",
		"evaluation_node       = risk_perception.evaluation_node:main",
		"scan_cluster_detector = risk_perception.scan_cluster_detector_node:main",
		"coverage_mask         = risk_perception.coverage_mask_node:main",
        ],
    },
)


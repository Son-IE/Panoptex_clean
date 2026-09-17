from setuptools import find_packages, setup

package_name = 'risk_map_publisher'
setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User A',
    maintainer_email='user-a@example.com',
    description='Synthetic and model-driven OccupancyGrid risk-map publishers.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'synthetic_risk_publisher = risk_map_publisher.synthetic_risk_publisher:main',
    ]},
)

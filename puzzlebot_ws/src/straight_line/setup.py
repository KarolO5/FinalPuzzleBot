from setuptools import setup

package_name = 'straight_line'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'odometry    = straight_line.odometry:main',
            'pd_controller = straight_line.pd_controller:main',
            'line_follower_cv      = line_follower.line_follower_cv:main',
        ],
    },
)
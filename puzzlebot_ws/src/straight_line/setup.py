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
            'line_follower_cv = straight_line.line_follower_cv:main',
            'camera_node      = straight_line.camera_node:main',  
            'semaforo         = straight_line.semaforo:main',
            #'sign_detector  = straight_line.sign_detector:main', # cuando esté listo
        ],
    },
)
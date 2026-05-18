FROM ros:humble AS base

RUN apt-get update && apt-get install -y \
    python3-pip \
    cmake \
    git \
    python3-colcon-common-extensions \
    ros-humble-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# ── micro-ROS Agent ────────────────────────────────────────────────
RUN mkdir -p /uros_ws/src
WORKDIR /uros_ws

RUN git clone -b humble https://github.com/micro-ROS/micro_ros_msgs.git src/micro_ros_msgs && \
    git clone -b humble https://github.com/micro-ROS/micro-ROS-Agent.git src/micro_ros_agent

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --packages-select micro_ros_msgs && \
    colcon build --packages-select micro_ros_agent"

# ── Workspace del robot ────────────────────────────────────────────
COPY puzzlebot_ws/ /puzzlebot_ws/
WORKDIR /puzzlebot_ws

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    source /uros_ws/install/setup.bash && \
    rm -rf build install log && \
    colcon build --packages-select puzzlebot_msgs straight_line"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
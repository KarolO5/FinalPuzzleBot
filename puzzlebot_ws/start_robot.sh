#!/bin/bash
# =============================================================================
# start_robot.sh — Lanza todos los nodos del PuzzleBot
# =============================================================================
# Uso:
#   chmod +x start_robot.sh
#   ./start_robot.sh
#
# Ctrl+C detiene todos los nodos lanzados.
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] $1${NC}"; }

# ── Sources ───────────────────────────────────────────────────────────────────
log "Sourcing ROS Jazzy..."
source /opt/ros/jazzy/setup.bash

log "Sourcing micro-ROS workspace..."
source ~/uros_ws/install/setup.bash

log "Sourcing PuzzleBot workspace..."
cd ~/puzzlebot_docker/puzzlebot_ws
source install/setup.bash

# ── Limpieza al salir (Ctrl+C) ────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    warn "Deteniendo todos los nodos..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done
    log "Todos los nodos detenidos."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Función para lanzar nodo en background ────────────────────────────────────
launch() {
    local name="$1"
    shift
    log "Lanzando: $name"
    "$@" &
    PIDS+=($!)
    sleep 4
}

# =============================================================================
# ORDEN DE LANZAMIENTO
# =============================================================================

# 1. micro-ROS Agent (puente USB ↔ ROS 2)
launch "micro_ros_agent" \
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200

# 2. Odometría
launch "odometry" \
    ros2 run straight_line odometry

# 3. Cámara
launch "camera_node" \
    ros2 run straight_line camera_node

# 4. Seguidor de línea
launch "line_follower_cv" \
    ros2 run straight_line line_follower_cv

# =============================================================================
log "Todos los nodos activos. Ctrl+C para detener."
log "PIDs: ${PIDS[*]}"
echo ""
warn "Fotos guardándose en: $HOME/puzzlebot_evidence/line_follower/frames/"

# Esperar indefinidamente hasta Ctrl+C
wait

#!/bin/bash

# Script: GPU Temperature Monitor
# Description: Monitor NVIDIA GPU temperature and performance metrics
# Author: Ozkan
# Date: 2024

# Configuration
TEMP_THRESHOLD=80
MEMORY_THRESHOLD=90
POLL_INTERVAL=2
LOG_DIR="./gpu_logs"

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stopping monitoring...${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Create log directory
mkdir -p "$LOG_DIR"
check_nvidia() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi command not found. No NVIDIA GPU detected.${NC}"
        exit 1
    fi
}

# Print section header
print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
}

# Get GPU info
get_gpu_info() {
    print_header "GPU Name and Temperature"
    nvidia-smi --query-gpu=gpu_name,temperature.gpu --format=csv || echo "Failed to get GPU info"
}

# Get system resources
get_system_resources() {
    print_header "CPU and Memory Usage"
    top -bn1 | grep "Cpu(s)" && top -bn1 | grep "MiB Mem" || echo "Failed to get CPU/Memory info"
    
    print_header "Swap Memory and CUDA Version"
    top -bn1 | grep "MiB Swap" && nvidia-smi | grep "CUDA Version" || echo "Failed to get Swap/CUDA info"
}

# Get GPU process info
get_gpu_processes() {
    print_header "GPU Processes"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || echo "Failed to get GPU processes"
}

# Get GPU utilization
get_gpu_utilization() {
    print_header "GPU Utilization and Memory Usage"
    paste <(nvidia-smi --query-gpu=utilization.gpu --format=csv) \
          <(nvidia-smi --query-gpu=memory.used --format=csv) || echo "Failed to get GPU utilization"
}

# Get GPU power and clock
get_gpu_power() {
    print_header "GPU Power Draw and Clock Speed"
    paste <(nvidia-smi --query-gpu=power.draw --format=csv) \
          <(nvidia-smi --query-gpu=clocks.gr --format=csv) || echo "Failed to get GPU power/clock"
}

# Monitor GPU fan speed
get_fan_speed() {
    print_header "GPU Fan Speed"
    nvidia-smi --query-gpu=fan.speed --format=csv,noheader || echo "Failed to get fan speed"
}

# Check temperature thresholds
check_temp_threshold() {
    local TEMP_THRESHOLD=${1:-80}
    local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    if [ "$temp" -gt "$TEMP_THRESHOLD" ]; then
        echo -e "${RED}Warning: GPU temperature above ${TEMP_THRESHOLD}°C${NC}"
        notify-send "GPU Temperature Warning" "Temperature above ${TEMP_THRESHOLD}°C" 2>/dev/null || true
    fi
}

# Get detailed memory stats
get_memory_stats() {
    print_header "GPU Memory Statistics"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv || echo "Failed to get memory stats"
}

# Monitor throttling
check_throttling() {
    print_header "GPU Throttling Status"
    nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv || echo "Failed to get throttling info"
}

# Continuous monitoring mode
monitor_continuous() {
    print_header "Continuous Monitoring (Ctrl+C to stop)"
    local count=0
    while true; do
        clear
        get_gpu_info
        get_fan_speed
        check_temp_threshold "$TEMP_THRESHOLD"
        get_memory_stats
        check_throttling
        
        # Log every 5 minutes
        if [ $((count % 150)) -eq 0 ]; then
            log_performance
        fi
        
        count=$((count + 1))
        sleep "$POLL_INTERVAL"
    done
}

# Log performance metrics
log_performance() {
    local log_file="$LOG_DIR/gpu_metrics_$(date +%Y%m%d_%H%M%S).log"
    print_header "Logging Performance to $log_file"
    {
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        get_gpu_info
        get_gpu_utilization
        get_memory_stats
        get_gpu_power
        echo "----------------------------------------"
    } >> "$log_file"
    echo "Metrics logged to $log_file"
}

main() {
    check_nvidia
    
    case "$1" in
        "--monitor") monitor_continuous ;;
        "--log") log_performance ;;
        "--threshold") check_temp_threshold "$2" ;;
        "--help")
            echo "Usage: $0 [--monitor|--log|--threshold <temp>|--help]"
            echo "  --monitor    : Continuous monitoring mode"
            echo "  --log       : Log current metrics"
            echo "  --threshold : Set temperature threshold"
            echo "  --help      : Show this help"
            ;;
        *)
            get_gpu_info
            get_system_resources
            get_gpu_processes
            get_gpu_utilization
            get_gpu_power
            get_fan_speed
            get_memory_stats
            check_throttling
            check_temp_threshold
            ;;
    esac
}

main "$@"
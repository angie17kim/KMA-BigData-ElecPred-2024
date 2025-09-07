#!/bin/bash

# nvidia-smi 명령어를 실행하여 python3 프로세스의 PID를 추출
PYTHON_PIDS=$(nvidia-smi | grep 'python3' | awk '{print $5}')

# ps 명령어를 실행하여 dask 관련 프로세스의 PID를 추출
DASK_PIDS=$(ps aux | grep 'dask' | grep -v grep | awk '{print $2}')

# 프로세스 상태를 확인하는 함수
get_process_state() {
    PID=$1
    STATE=$(ps -o stat= -p $PID)
    if [[ "$STATE" == *"Z"* ]]; then
        echo "Zombie"
    elif [[ "$STATE" == *"T"* ]]; then
        echo "Stopped"
    else
        echo "Running"
    fi
}

# 추출된 PID를 확인하고 소유주가 'minchan'인 경우에만 종료
terminate_processes() {
    local PIDS=$1
    local PROCESS_NAME=$2
    for PID in $PIDS; do
        if [ -n "$PID" ]; then
            OWNER=$(ps -o user= -p $PID)
            if [ "$OWNER" = "minchan" ]; then
                STATE=$(get_process_state $PID)
                if [ "$STATE" = "Zombie" ] || [ "$STATE" = "Stopped" ]; then
                    echo "Terminating $PROCESS_NAME process with PID: $PID (Owner: $OWNER, State: $STATE)"
                    kill -9 $PID
                else
                    echo "Skipping $PROCESS_NAME process with PID: $PID (Owner: $OWNER, State: $STATE)"
                fi
            else
                echo "Skipping $PROCESS_NAME process with PID: $PID (Owner: $OWNER)"
            fi
        fi
    done
}

# python3 프로세스 종료
terminate_processes "$PYTHON_PIDS" "python3"

# dask 프로세스 종료
terminate_processes "$DASK_PIDS" "dask"

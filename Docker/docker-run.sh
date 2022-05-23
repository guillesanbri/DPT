[[ $# -eq 0 ]] && { echo "Usage: $0 tag"; exit 1; }

wandb docker-run --gpus all --rm -it --name ${1}_container --ipc=host -p 8888:8888 -v $(pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY $1

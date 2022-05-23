[[ $# -eq 0 ]] && { echo "Usage: $0 tag"; exit 1; }

wandb docker-run -d --gpus all --rm -it --name ${1}_container --ipc=host -p 8888:8888 -v $(pwd):/workspace  $1

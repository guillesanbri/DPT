#!/bin/bash
# [[ $# -eq 0 ]] && { echo "Usage: $0 tag"; exit 1; }

rm output_monodepth/*.{png,pfm}
python run_monodepth.py --model_type dpt_hybrid_kitti --kitti_crop --absolute_depth
python ./eval_with_pngs.py --pred_path ./output_monodepth/ --gt_path ./input/gt/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --garg_crop --do_kb_crop

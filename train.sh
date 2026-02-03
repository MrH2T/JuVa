torchrun --nproc_per_node=7 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 128 --noise_scale 1.0 \
--batch_size 8 --blr 1e-5 \
--epochs 4000 --warmup_epochs 5 \
--gen_bsz 8 \
--num_sampling_steps 50 \
--cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./model --resume ./model \
--data_path ./maze3x3 \
--online_eval --eval_freq 50  # 添加这两行：开启在线评估，每50轮出一个Demo
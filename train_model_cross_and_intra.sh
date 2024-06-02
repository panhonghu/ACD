python train.py --data_path /home/ubuntu/data/honghu/D2D/data \
                --model_path ./save_model \
                --log_path ./log \
                --diffusion_only_epoch 50 \
                --total_epoch 70 \
                --save_epoch 5 \
                --in_channels 4 \
                --gpu 1 \
                --batch_size 3 \
                --img_h 144 \
                --img_w 72 \
                --num_diffusion_timesteps 1000 \
                --timesteps 20 \
                --g_steps 1 \
                --d_steps 2 \
                --model_path ./save_model/ \
                --model_prefix Diffusion-cross_and_intra \
                --log_file log-Diffusion-cross_and_intra.txt



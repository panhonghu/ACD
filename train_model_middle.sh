python train.py --data_path /home/ubuntu/data/honghu/D2D/data/ \
                --model_path ./save_model \
                --middle \
                --log_path ./log \
                --total_epoch 50 \
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
                --model_prefix Diffusion-middle \
                --log_file log-Diffusion-middle.txt



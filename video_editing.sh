python scripts/video_editing.py \
    --input_path assets/car-turn.mp4 \
    --output_path results/car.gif \
    --prompt "a red car is turning" \
    --n_steps 50 \
    --strength 0.7 \

# python scripts/video_editing.py \
#     --input_path assets/tiger.mp4 \
#     --output_path results/tiger_cartoon.gif \
#     --prompt "a tiger in cartoon style" \
#     --n_steps 50 \
#     --strength 0.7 \
#     --n_frames 8 \
#     --guidance_scale 7.5

# python scripts/video_editing.py \
#     --input_path assets/hike.mp4 \
#     --output_path results/hike.gif \
#     --prompt "A man hikes in forest" \
#     --n_steps 50 \
#     --strength 0.7 \
#     --n_frames 8 \
#     --guidance_scale 7.5

# python scripts/video_editing.py \
#     --input_path assets/swan.mp4 \
#     --output_path results/swan.gif \
#     --prompt "A swan in ink painting style" \
#     --n_steps 50 \
#     --strength 0.8 \
#     --n_frames 8 \
#     --guidance_scale 7.5

# Comparison Experiment for original Stable Diffusion 1.5

# python scripts/video_editing.py \
#     --input_path assets/car-turn.mp4 \
#     --output_path results/car_noaf.gif \
#     --prompt "a red car is turning" \
#     --n_prompt "" \
#     --n_steps 50 \
#     --strength 0.7 \
#     -no_af 
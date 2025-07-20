exp=3d_diffuser_actor

tasks=(
    setup_checkers screw_nail insert_usb_in_computer plug_charger_in_power_supply unplug_charger put_umbrella_in_umbrella_stand insert_onto_square_peg
)
data_dir=data/precise/test/
num_episodes=100
gripper_loc_bounds_file=./tasks/7_precise_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="left_shoulder,right_shoulder,wrist,front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=1
checkpoint=./train_logs/Flow/flowram-C120-B6-lr1e-4-DI1-2-H3-DT100/200000.pth
quaternion_format=xyzw  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1  CUDA_VISIBLE_DEVICES=5 python evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 32 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file ecal_DDIM/32/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions data/RLbench_Peract/instructions/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 20 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1
done


# given a list of string, memorize them and print them out

# list_id=("union-timesteps")

# # run command
# for i in ${list_id[@]}; do
#     # python modularity/wanda.py memorize memorize_$i
#     python benchmarks/i2p_eval.py --gpu 6 --fine_tuned_unet $i --dataset_type ring-a-bell
# done

python benchmarks/i2p_eval.py --gpu 0 --fine_tuned_unet union-timesteps --dataset_type ring-a-bell --model_id runwayml/stable-diffusion-v1-5
python benchmarks/i2p_eval.py --gpu 6 --fine_tuned_unet 'CompVis/stable-diffusion-v1-4-safe' --dataset_type ring-a-bell
python benchmarks/i2p_eval.py --gpu 6 --fine_tuned_unet 'CompVis/stable-diffusion-v1-4-safe' --dataset_type mma
python benchmarks/i2p_eval.py --gpu 6 --fine_tuned_unet 'CompVis/stable-diffusion-v1-4-safe' --dataset_type i2p
python benchmarks/i2p_eval.py --gpu 6 --fine_tuned_unet "stabilityai/stable-diffusion-2" --dataset_type mma
python benchmarks/i2p_eval.py --gpu 6 --fine_tuned_unet "stabilityai/stable-diffusion-2" --dataset_type ring-a-bell
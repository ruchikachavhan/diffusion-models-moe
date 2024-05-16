# given a list of string, memorize them and print them out

list_id=(10 11 12 13 14 15 16 17 18 19)

# run command
for i in ${list_id[@]}; do
    # python benchmarks/inference_mem.py --skill_method AP --fine_tuned_unet 'union-timesteps' --concepts_to_remove memorize_$i --gpu 1
    # python modularity/wanda.py memorize memorize_$i
    # python modularity/skilled_neuron_ap.py --concepts_to_remove memorize_$i --gpu 6 --fine_tuned_unet union-timesteps
    python benchmarks/save_union_over_ap.py --concept memorize_$i --timesteps 20 --select_ratio 0.0
done
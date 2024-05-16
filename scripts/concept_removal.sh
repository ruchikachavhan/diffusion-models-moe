# make a list of artist names
# concepts=("Salvador Dali")

# # remove the concept from the list
# for string in "${concepts[@]}"; do
#     # execute command
#     python benchmarks/eval_coco.py  --dataset_type concept_removal --fine_tuned_unet union-timesteps --concepts_to_remove "$string" --gpu 1
# done

python benchmarks/artist_removal.py --concepts_to_remove 'Van Gogh' --fine_tuned_unet 'uce' --gpu 0
python benchmarks/artist_removal.py --concepts_to_remove 'Monet' --fine_tuned_unet 'uce' --gpu 0
python benchmarks/artist_removal.py --concepts_to_remove 'Pablo Picasso' --fine_tuned_unet 'uce' --gpu 0
python benchmarks/artist_removal.py --concepts_to_remove 'Salvador Dali' --fine_tuned_unet 'uce' --gpu 0
python benchmarks/artist_removal.py --concepts_to_remove 'Leonardo Da Vinci' --fine_tuned_unet 'uce' --gpu 0
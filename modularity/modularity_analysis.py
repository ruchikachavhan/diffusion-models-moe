import json
import os
import sys
import tqdm
from mod_utils import get_prompts
sys.path.append('moefication')
from helper import modify_ffn_to_experts
sys.path.append(os.getcwd())
import utils
from neuron_receivers import NeuronPredictivity, NeuronPredictivityBB, ExpertPredictivity

def get_neuron_receivers(args, num_geglu):
     # Neuron receiver with forward hooks to measure predictivity
    if args.modularity['predictvity_for'] == 'expert':
        neuron_receiver = ExpertPredictivity
    else:
        if args.modularity['bounding_box']:
            neuron_receiver = NeuronPredictivityBB
        else:
            neuron_receiver = NeuronPredictivity
    neuron_pred_base = neuron_receiver(args.seed, args.timesteps, 
                                       num_geglu, replace_fn = args.replace_fn,
                                       keep_nsfw = args.modularity['keep_nsfw'])
    neuron_pred_adj = neuron_receiver(args.seed, args.timesteps, num_geglu, replace_fn = args.replace_fn,
                                      keep_nsfw = args.modularity['keep_nsfw'])
    
    return neuron_pred_base, neuron_pred_adj

def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
    args.configure('modularity')

    # Model
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    args.replace_fn = replace_fn
    print("Replce fn: ", replace_fn)
    model = model.to(args.gpu)

    if args.modularity['predictvity_for'] == 'expert':
        args.moefication = {}
        args.moefication['topk_experts'] = 1.0
        model, ffn_names_list, num_experts_per_ffn = modify_ffn_to_experts(model, args)
    
    
    # Neuron receiver
    neuron_pred_base, neuron_pred_adj = get_neuron_receivers(args, num_geglu)
   
    # Test the model
    if args.fine_tuned_unet is not None:
        neuron_pred_base.test(model)
        neuron_pred_adj.test(model)
        print("Neuron receiver tests passed")

    base_prompts, adj_prompts, _ = get_prompts(args)

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)

    iter = 0

    # initialise a standard deviation mesurement for difference in predictivities for concept and base prompt
    diff_std = {}
    for l in range(num_geglu):
        diff_std[l] = utils.StandardDev()

    for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
        if iter >= 5 and args.dbg:
            break
        print("text: ", ann, ann_adj)
        
        neuron_pred_base.reset_layer()
        out, _ = neuron_pred_base.observe_activation(model, ann,
                                                    bboxes=bb_coordinates_layer_base[ann] if args.modularity['bounding_box'] else None)

        neuron_pred_adj.reset_layer()
        out_adj, _ = neuron_pred_adj.observe_activation(model, ann_adj, 
                                                    bboxes=bb_coordinates_layer_adj[ann_adj] if args.modularity['bounding_box'] else None)

        for l in range(num_geglu):
            print(neuron_pred_base.max_gate[l], neuron_pred_adj.max_gate[l])
            diff_std[l].update(neuron_pred_base.max_gate[l] - neuron_pred_adj.max_gate[l])

        # save images
        out.save(os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
        # out_adj.save(os.path.join(args.modularity['img_save_path'], f'adj_{iter}.jpg'))
        iter += 1


    for l in range(num_geglu):
        diff_std[l] = diff_std[l].stddev().tolist()  
        neuron_pred_adj.predictivity[l] = neuron_pred_adj.predictivity[l].avg.tolist()
        neuron_pred_base.predictivity[l] = neuron_pred_base.predictivity[l].avg.tolist()
        print(f"Layer {l} predictivity for base: {len(neuron_pred_base.predictivity[l])}")
        print(f"Layer {l} predictivity for adj: {len(neuron_pred_adj.predictivity[l])}")

    # save results
    print("Saving results")
    save_type = 'bb' if args.modularity['bounding_box'] else ''
    save_type = save_type + '_expert' if args.modularity['predictvity_for'] == 'expert' else save_type
    print(save_type)
    with open(os.path.join(args.save_path, f'predictivity_adj{save_type}.json'), 'w') as f:
        json.dump(neuron_pred_adj.predictivity, f)
    with open(os.path.join(args.save_path, f'predictivity_base{save_type}.json'), 'w') as f:
        json.dump(neuron_pred_base.predictivity, f)
    # # save diff_std
    with open(os.path.join(args.save_path, f'diff_std{save_type}.json'), 'w') as f:
        json.dump(diff_std, f)


if __name__ == "__main__":
    main()

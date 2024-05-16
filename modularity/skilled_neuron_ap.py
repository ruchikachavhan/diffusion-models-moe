import json
import os
import sys
import tqdm
from mod_utils import get_prompts
sys.path.append('moefication')
from helper import modify_ffn_to_experts
sys.path.append(os.getcwd())
import utils
import numpy as np
from neuron_receivers import NeuronPredictivity, NeuronPredictivityBB, ExpertPredictivity

non_memorised = open('modularity/datasets/non_mem.txt', 'r').read().split('\n')


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
    # Input topk experts for evaluation
    concept = str(sys.argv[1])
    if concept is not None:
        args.modularity['adjective'] = str(concept).strip()
    print(f"Adjective: {args.modularity['adjective']}")

    # check if a second argument exists
    if len(sys.argv) > 2:
        args.modularity['file_name'] = str(sys.argv[2])
        args.modularity['adjective'] = str(sys.argv[2])

    
    print(f"Adjective: {args.modularity['adjective']}")
    print(f"File name: {args.modularity['file_name']}")
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
    labels = [1] * len(adj_prompts)
    adj_prompts += non_memorised
    base_prompts = [" "] * len(adj_prompts)
    labels += [0] * len(non_memorised)

    print("prompts: ", adj_prompts)

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)

    iter = 0

    # initialise a standard deviation mesurement for difference in predictivities for concept and base prompt
    diff_std = {}
    for t in range(args.timesteps):
        diff_std[t] = {}
        for l in range(args.n_layers):
            diff_std[t][l] = 0

    print(args.modularity['skill_neuron_path'])
    if not os.path.exists(os.path.join(args.modularity['skill_neuron_path'], 'diff_std.json')):
        for ann, ann_adj, label in tqdm.tqdm(zip(base_prompts, adj_prompts, labels)):
            if iter >= 5 and args.dbg:
                break
            print("text: ", ann, ann_adj, label)
            
            # sample random seed and set it for both neuron_pred_base and neuron_pred_adj
            # seed = np.random.randint(0, 100000)
            # neuron_pred_adj.seed = seed
            # neuron_pred_base.seed = seed

            neuron_pred_base.reset_time_layer()
            out, _ = neuron_pred_base.observe_activation(model, ann,
                                                        bboxes=bb_coordinates_layer_base[ann] if args.modularity['bounding_box'] else None)

            neuron_pred_adj.reset_time_layer()
            # ann_adj = ann_adj.split('\n')[0]
            out_adj, _ = neuron_pred_adj.observe_activation(model, ann_adj, 
                                                        bboxes=bb_coordinates_layer_adj[ann_adj] if args.modularity['bounding_box'] else None)

            for t in range(args.timesteps):
                for l in range(args.n_layers):
                    diff = neuron_pred_base.max_gate[t][l] < neuron_pred_adj.max_gate[t][l]
                    diff = diff.astype(int)
                    diff = (diff == label)
                    diff_std[t][l] += diff

                # save images
            out.save(os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
            out_adj.save(os.path.join(args.modularity['img_save_path'], f'adj_{iter}.jpg'))
            iter += 1

        # convert all diff to list before saving
        for t in range(args.timesteps):
            for l in range(args.n_layers):
                # take average of predictivity
                diff_std[t][l] = diff_std[t][l]/iter
                diff_std[t][l] = diff_std[t][l].tolist()

        # save diff_std in a file
        with open(os.path.join(args.modularity['skill_neuron_path'], 'diff_std.json'), 'w') as f:
            json.dump(diff_std, f)
    else:
        with open(os.path.join(args.modularity['skill_neuron_path'], 'diff_std.json'), 'r') as f:
            diff_std = json.load(f)
            # print("Loaded diff_std: ", diff_std)

    args.modularity['skill_neuron_path'] = args.modularity['skill_neuron_path'].replace(str(args.modularity['condition']['skill_ratio']), '0.05')
    if not os.path.exists(args.modularity['skill_neuron_path']):
        os.makedirs(args.modularity['skill_neuron_path'])
    print("Skill neuron path: ", args.modularity['skill_neuron_path'])
    predictivity = {}
    print(diff_std.keys())
    # convert dict keys to ints 
    diff_std = {int(k): v for k, v in diff_std.items()}
    for t in range(args.timesteps):
        diff_std[t] = {int(k): v for k, v in diff_std[t].items()}
    for t in range(args.timesteps):
        predictivity[t] = {}
        for l in range(args.n_layers):
            diff_std[t][l] = np.array(diff_std[t][l])
            print("Time step: ", t, "Layer: ", l, "Number of skilled neurons: ", diff_std[t][l])
            # take max of a and 1-a of each element of diff_std
            # pred = np.maximum(diff_std[t][l], 1 - diff_std[t][l])
            # save the value of 
            pred = diff_std[t][l]
            print(pred)

            # sort predictivity and select indices of top k
            pred_indices = pred.argsort()
            pred_indices = pred_indices[::-1]
            k = int(0.05 * len(pred))
            indices = pred_indices[:k]
            # make a binary mask and set indeices to 1
            mask = np.zeros(len(pred))
            mask[indices] = 1
            print("Predictivity: ", np.mean(mask))
            # save the binary mask in a json file
            print(os.path.join(args.modularity['skill_neuron_path'], f'predictivity_{t}_{l}.json'))
            with open(os.path.join(args.modularity['skill_neuron_path'], f'predictivity_{t}_{l}.json'), 'w') as f:
                json.dump(mask.tolist(), f)

    # # save results
    # print("Saving results")
    # save_type = 'bb' if args.modularity['bounding_box'] else ''
    # save_type = save_type + '_expert' if args.modularity['predictvity_for'] == 'expert' else save_type
    # print(save_type)
    # print(neuron_pred_adj.predictivity)
    # neuron_pred_adj.predictivity.save(os.path.join(args.save_path, f'predictivity_adj{save_type}.json'))
    # neuron_pred_base.predictivity.save(os.path.join(args.save_path, f'predictivity_base{save_type}.json'))
    # # save diff_std
    # with open(os.path.join(args.save_path, f'diff_std{save_type}.json'), 'w') as f:
    #     json.dump(diff_std, f)


if __name__ == "__main__":
    main()

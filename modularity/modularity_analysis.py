import json
import os
import sys
import tqdm
sys.path.append(os.getcwd())
import utils
from neuron_receivers import NeuronPredictivity, NeuronPredictivityBB


def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
    args.configure('modularity')

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks to measure predictivity
    if args.modularity['bounding_box']:
        neuron_pred_base = NeuronPredictivityBB(args.seed, args.timesteps, num_geglu)
        neuron_pred_adj = NeuronPredictivityBB(args.seed, args.timesteps, num_geglu)
    else:
        neuron_pred_base = NeuronPredictivity(args.seed, args.timesteps, num_geglu)
        neuron_pred_adj = NeuronPredictivity(args.seed, args.timesteps, num_geglu)

    # Test the model
    if args.fine_tuned_unet is not None:
        neuron_pred_base.test(model)
        neuron_pred_adj.test(model)
        print("Neuron receiver tests passed")

    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    base_prompts = [f'a {thing.strip()}' for thing in objects]
    # add an adjective of choice to every element in things list
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in objects]

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
            print(bb_coordinates_layer_adj.keys())
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)
            print(bb_coordinates_layer_base.keys())

    iter = 0

    # initialise a standard deviation mesurement for difference in predictivities for concept and base prompt
    diff_std = {}
    for t in range(args.timesteps):
        diff_std[t] = {}
        for l in range(args.n_layers):
            diff_std[t][l] = utils.StandardDev()

    for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
        if iter >= 5 and args.dbg:
            break
        print("text: ", ann, ann_adj)
        
        neuron_pred_base.reset_time_layer()
        out, _ = neuron_pred_base.observe_activation(model, ann, bboxes=bb_coordinates_layer_base[ann] if args.modularity['bounding_box'] else None)

        neuron_pred_adj.reset_time_layer()
        out_adj, _ = neuron_pred_adj.observe_activation(model, ann_adj, bboxes=bb_coordinates_layer_adj[ann_adj] if args.modularity['bounding_box'] else None)

        for t in range(args.timesteps):
            for l in range(args.n_layers):
                diff = neuron_pred_base.max_gate[t][l] - neuron_pred_adj.max_gate[t][l]
                diff_std[t][l].update(diff)
        # save images
        out.save(os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
        out_adj.save(os.path.join(args.modularity['img_save_path'], f'adj_{iter}.jpg'))
        iter += 1

    # save results
    print("Saving results")
    if args.modularity['bounding_box']:
        neuron_pred_adj.predictivity.save(os.path.join(args.save_path, 'predictivity_adj_bb.json'))
        neuron_pred_base.predictivity.save(os.path.join(args.save_path, 'predictivity_base_bb.json'))
    else:
        neuron_pred_adj.predictivity.save(os.path.join(args.save_path, 'predictivity_adj.json'))
        neuron_pred_base.predictivity.save(os.path.join(args.save_path, 'predictivity_base.json'))
        # save diff_std
        for t in range(args.timesteps):
            for l in range(args.n_layers):
                diff_std[t][l] = diff_std[t][l].stddev().tolist()
        
        # save diff_std
        with open(os.path.join(args.save_path, 'diff_std.json'), 'w') as f:
            json.dump(diff_std, f)




if __name__ == "__main__":
    main()

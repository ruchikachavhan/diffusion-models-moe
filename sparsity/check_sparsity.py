import json
import os
import sys
import tqdm
sys.path.append(os.getcwd())
from utils import get_sd_model, coco_dataset, Config, StatMeter
from neuron_receivers import SparsityMeasure

def main():
    args = Config('experiments/config.yaml', 'sparsity')
    args.configure('sparsity')
    # Model
    model, num_geglu = get_sd_model(args)
    model = model.to(args.gpu)
    # Eval dataset
    imgs, anns = coco_dataset(args.dataset['path'], 'val', args.inference['num_images'])

    # Neuron receiver to store gates for every sample
    neuron_receiver = SparsityMeasure(args.seed)
    if args.fine_tuned_unet is not None:
        neuron_receiver.test(model)
        print("Neuron receiver test passed")

    iter = 0
    results = StatMeter(T=args.timesteps, n_layers=num_geglu)
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 5 and args.dbg:
            break
        print("Iter: ", iter)
        print("text: ", ann)
        out, gates = neuron_receiver.observe_activation(model, ann)
        
        # divide gate into chunks of number of time steps
        for i in range(0, len(gates), num_geglu):
            gate_timestep = gates[i:i+num_geglu]
            for j, gate in enumerate(gate_timestep):
                if j > num_geglu:
                    continue
                # check sparsity
                # check if values of the gate == 0
                mask = gate == 0.0
                # % of neurons that are 0 out of total neurons (= hidden dimension)
                exact_zero_ratio = mask.int().sum(-1).float() / gate.shape[-1]
                # Take mean over all tokens
                exact_zero_ratio = exact_zero_ratio.mean()
                results.update(exact_zero_ratio.item(), i//num_geglu, j)
        iter += 1

    print(f'Saving results to {args.save_path}')
    results.save(os.path.join(args.save_path, 'sparsity.json'))

        
    
if __name__ == '__main__':
    main()


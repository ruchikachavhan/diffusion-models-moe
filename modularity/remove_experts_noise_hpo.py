import json
import os
import sys
import torch
import gc
import numpy as np
from PIL import Image, ImageFilter
from mod_utils import get_prompts, LLAVAScorer
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import RemoveNeuronsNoiseHPO, RemoveExperts, BaseUNetReceiver
sys.path.append('moefication')
from helper import modify_ffn_to_experts
from PIL import ImageDraw, ImageFont
from paired_t_test import critical_value_ranges
import optuna
from optuna.trial import TrialState

 
args = utils.Config('experiments/remove_skills.yaml', 'modularity')
args.configure('modularity')
# change remove_neurons_path
args.modularity['remove_neuron_path'] = os.path.join(args.modularity['remove_neuron_path'], 'noise_hpo_iterations')
args.modularity['remove_neuron_path_val'] = os.path.join(args.modularity['remove_neuron_path_val'], 'noise_hpo_iterations')
if not os.path.exists(args.modularity['remove_neuron_path']):
    os.makedirs(args.modularity['remove_neuron_path'])
if not os.path.exists(args.modularity['remove_neuron_path_val']):
    os.makedirs(args.modularity['remove_neuron_path_val'])

# if msfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_experts(adj_prompts, model, base_receiver, neuron_remover, args, bounding_box, save_path, base_prompts):
    iter = 0
    noise_diff = {}
    for t in range(args.timesteps):
        noise_diff[t] = utils.Average()

    #  save noise for every sample in the prompt list
    noise_diff_per_sample = {}

    for ann, ann_adj in zip(base_prompts, adj_prompts):
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann, ann_adj)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Get the image for concept prompt 
        out = model(ann_adj).images[0]

        # Forward pass for the base prompt and collect unet outputs
        # Example of base_prompt - "A photo of a cat"
        base_receiver.reset_time()
        # ann_adj = ann_adj + '\n'
        out_base, _ = base_receiver.observe_activation(model, ann, bboxes=bounding_box[ann_adj] if bounding_box is not None else None)
        unet_outputs = base_receiver.unet_output
        # Forward pass for the adjective prompt and collect unet outputs
        # Example of adj_prompt - "A photo of a cat in Van Gogh Style"
        neuron_remover.reset_time_layer()
        # ann_adj = ann_adj + '\n'
        out_adj, _ = neuron_remover.observe_activation(model, ann_adj, bboxes=bounding_box[ann_adj] if bounding_box is not None else None)
        unet_outputs_adj = neuron_remover.unet_output

        # calculate the difference (L2 norm) of the noise
        # We want to minimise this difference because we want the noise to be similar for the prompt and adjective propmt after removing neurons
        noise_diff_per_sample[iter] = 0.0
        for t in range(args.timesteps):
            diff = torch.norm(unet_outputs[t] - unet_outputs_adj[t]).item()
            noise_diff[t].update(diff)
            # l1 norm
            u1 = unet_outputs[t].view(-1)
            u2 = unet_outputs_adj[t].view(-1)
            # normalise
            u1 = torch.nn.functional.normalize(u1, dim=0, p=1)
            u2 = torch.nn.functional.normalize(u2, dim=0, p=1)
            # take norm of difference
            noise_diff_per_sample[iter] += torch.norm(u1 - u2).item()
        
        
        noise_diff_per_sample[iter] /= args.timesteps

        # Save images
        # stitch the images to keep them side by side
        out = out.resize((256, 256)) # This is concept image (without removal)
        out_adj = out_adj.resize((256, 256)) # This is concept image after removal
        out_base = out_base.resize((256, 256)) # This is base image (without removal)
        # make bigger image to keep both images side by side with white space in between
        new_im = Image.new('RGB', (530+256, 290))

        if args.modularity['keep_nsfw']:
            out = blur_image(out, args.modularity['condition']['is_nsfw'])
            
        new_im.paste(out, (0,40))
        new_im.paste(out_adj, (275,40))
        new_im.paste(out_base, (530,40))

        # write the prompt on the image
        draw = ImageDraw.Draw(new_im)
        font = ImageFont.load_default(size=15)
        draw.text((80, 15), ann_adj, (255, 255, 255), font=font)
        draw.text((350, 15), 'w/o experts', (255, 255, 255), font=font)

        obj_name = base_prompts[iter].split(' ')[-1] if base_prompts is not None else ann_adj

        new_im.save(os.path.join(save_path, f'img_{iter}_{obj_name}.jpg'))

        # save images
        print("Image saved in ", save_path)
        if args.modularity['keep_nsfw']:
            out = blur_image(out, args.modularity['condition']['is_nsfw'])
            blurred_out_adj = blur_image(out_adj, args.modularity['condition']['is_nsfw'])
            blurred_out_adj.save(os.path.join(save_path, f'img_{iter}_adj_blurred.jpg'))
        out.save(os.path.join(save_path, f'img_{iter}.jpg'))
        out_adj.save(os.path.join(save_path, f'img_{iter}_adj.jpg'))
        iter += 1

    # calculate the average noise difference over all timesteps
    avg_noise_diff = 0.0
    for t in range(args.timesteps):
        avg_noise_diff += noise_diff[t].avg
    avg_noise_diff /= args.timesteps

    print("Average noise difference: ", avg_noise_diff)
    return avg_noise_diff, noise_diff_per_sample

def objective(trial):
    dof, conf_int, dof_critical_values_dict = critical_value_ranges()
    dof = [int(d) for d in dof]

    # model 
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    args.replace_fn = replace_fn
    model = model.to(args.gpu)
   
    # trial.suggest_something args are (name of the variable, range)
    # dof_trial_val = trial.suggest_categorical("dof", dof)
    # read object file
    # read file
    with open(os.path.join('modularity/datasets', args.modularity['file_name']+'.txt'), 'r') as f:
        objects = f.readlines()
    objects = [obj.strip() for obj in objects]

    dof_trial_val = len(objects) - 1
    print("---------- Starting Trial with parameters ----------", dof_trial_val)
    conf_trial_val = trial.suggest_categorical("conf_val", conf_int)
    print("---------- Starting Trial with parameters ----------", dof_trial_val, conf_trial_val)
    # trials_per_layer = []
    # for t in range(args.timesteps):
    #     trials_per_layer.append(trial.suggest_categorical(f"timestep_{t}", [0, 1]))
    base_receiver = BaseUNetReceiver(seed=args.seed, T=args.timesteps, n_layers=num_geglu)

    neuron_remover = RemoveNeuronsNoiseHPO(seed=args.seed, path_expert_indx = args.modularity['skill_neuron_path'],
                            T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=args.modularity['keep_nsfw'],
                            dof_val=dof_trial_val, conf_val=conf_trial_val, trials_per_layer=trial)
    
    adjectives = args.modularity['adjective']
    base_prompts, adj_prompts, _ = get_prompts(args)


    # llava_scorer = LLAVAScorer(objects, args.modularity['adjective'], args)

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
            print(bb_coordinates_layer_adj.keys())
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)

    root_path = args.modularity['remove_neuron_path'].split('remove_neurons')[0]
    save_path = os.path.join(root_path, f'dof_{dof_trial_val}_conf_{conf_trial_val}', 'noise_hpo_iterations', 'remove_neurons')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # remove experts
    score, noise_diff_per_sample = remove_experts(adj_prompts, model, base_receiver, neuron_remover, args,
                   bounding_box=bb_coordinates_layer_adj if args.modularity['bounding_box'] else None, 
                   save_path=save_path,
                   base_prompts=base_prompts)
    
    print("Noise difference per sample: ", noise_diff_per_sample)
    # save the noise difference for every sample for this trial in a json file
    with open(os.path.join(save_path, 'noise_diff_per_sample.json'), 'w') as f:
        json.dump(noise_diff_per_sample, f)

    
    # score to maximise
    # object detection score after removal + style score after removal
    # score = results['after_removal']['object_score'] + results['after_removal']['style_score']
    print("---------------- Trial results ----------------")
    print(f"DOF: {dof_trial_val}, Confidence interval: {conf_trial_val}")
    print(f"Score: {score}")

    torch.cuda.empty_cache()
    del model, neuron_remover, base_receiver
    gc.collect()
    return score


def main():
    # HPO trials
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    adjective = args.modularity['adjective']
    del args
    dof, conf_int, dof_critical_values_dict = critical_value_ranges()
    study = optuna.create_study(direction="minimize",
                                # storage="sqlite:///db.sqlite3", # Specify the storage URL here.
                                # study_name=f"trial-{adjective}-unet-noise-score-gridsearch-2", load_if_exists=True,
                                sampler=optuna.samplers.GridSampler({'conf_val': conf_int}))
    study.optimize(objective, n_trials=100, gc_after_trial=True)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # save best trial results
    if not os.path.exists('modularity/hpo_results'):
        os.makedirs('modularity/hpo_results')
    with open(f"modularity/hpo_results/{adjective}.json", 'w') as f:
        json.dump(trial.params, f)

if __name__ == "__main__":
    main()
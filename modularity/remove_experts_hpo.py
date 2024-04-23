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
from neuron_receivers import RemoveNeuronsHPO, RemoveExperts
sys.path.append('moefication')
from helper import modify_ffn_to_experts
from PIL import ImageDraw, ImageFont
from paired_t_test import critical_value_ranges
import optuna
from optuna.trial import TrialState

 
args = utils.Config('experiments/remove_skills.yaml', 'modularity')
args.configure('modularity')

# if msfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_experts(adj_prompts, model, neuron_receiver, args, bounding_box, save_path, base_prompts=None, llava_scorer=None):
    iter = 0

    for ann_adj in adj_prompts:
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann_adj)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # run model for the original text
        # out = model(ann_adj).images[0]
        if 'lcm' in args.model_id:
            out = model(ann_adj, num_inference_steps=4, guidance_scale=8.0).images[0]
        else:
            out = model(ann_adj).images[0]

        neuron_receiver.reset_time_layer()
        # ann_adj = ann_adj + '\n'
        out_adj, _ = neuron_receiver.observe_activation(model, ann_adj, bboxes=bounding_box[ann_adj] if bounding_box is not None else None)

        if llava_scorer is not None:
            obj_detect = llava_scorer.object_score(out, iter, before=True)
            print(f"Object detection label: {obj_detect}")
            obj_detect_after_removal = llava_scorer.object_score(out_adj, iter, before=False)
            print(f"Object detection label after removal: {obj_detect_after_removal}")

            # style scores
            style_before = llava_scorer.style_score(out, iter, label=2, before=True)
            print(f"Style score before removal: {style_before}")
            style_after = llava_scorer.style_score(out_adj, iter, label=1, before=False)
            print(f"Style score after removal: {style_after}")


        # stitch the images to keep them side by side
        out = out.resize((256, 256))
        out_adj = out_adj.resize((256, 256))
        # make bigger image to keep both images side by side with white space in between
        new_im = Image.new('RGB', (530, 290))

        if args.modularity['keep_nsfw']:
            out = blur_image(out, args.modularity['condition']['is_nsfw'])
            
        new_im.paste(out, (0,40))
        new_im.paste(out_adj, (275,40))

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
        torch.cuda.empty_cache()
    
    # print results of LLAVA scorer
    print("Results of LLAVA scorer:")
    results = llava_scorer.get_results()
    print(results)

    # save results
    with open(os.path.join(save_path, 'llava_results.json'), 'w') as f:
        json.dump(results, f)
    
    
    return results

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

    neuron_receiver = RemoveNeuronsHPO(seed=args.seed, path_expert_indx = args.modularity['skill_neuron_path'],
                            T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=args.modularity['keep_nsfw'],
                            dof_val=dof_trial_val, conf_val=conf_trial_val, trials_per_layer=trial)
    
    adjectives = args.modularity['adjective']
    base_prompts, adj_prompts, _ = get_prompts(args)


    llava_scorer = LLAVAScorer(objects, args.modularity['adjective'], args)

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
            print(bb_coordinates_layer_adj.keys())
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)

    root_path = args.modularity['remove_neuron_path'].split('remove_neurons')[0]
    save_path = os.path.join(root_path, f'dof_{dof_trial_val}_conf_{conf_trial_val}', 'remove_neurons')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # remove experts
    results = remove_experts(adj_prompts, model, neuron_receiver, args,
                   bounding_box=bb_coordinates_layer_adj if args.modularity['bounding_box'] else None, 
                   save_path=save_path,
                   base_prompts=base_prompts, 
                   llava_scorer=llava_scorer)
    
    # score to maximise
    # object detection score after removal + style score after removal
    # score = results['after_removal']['object_score'] + results['after_removal']['style_score']
    score = results['after_removal']['style_score']
    print("---------------- Trial results ----------------")
    print(f"DOF: {dof_trial_val}, Confidence interval: {conf_trial_val}")
    print(f"Score: {score}")

    torch.cuda.empty_cache()
    del model, llava_scorer, neuron_receiver
    gc.collect()
    return score


def main():
    # HPO trials
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    adjective = args.modularity['adjective']
    del args
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3", # Specify the storage URL here.
                                study_name=f"trial-{adjective}-styles-score-timesteps>10-per-sample", load_if_exists=True)
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
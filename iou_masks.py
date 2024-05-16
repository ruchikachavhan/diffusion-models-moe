# read files which are binary matrices and take pairwise IOU
import json
import numpy as np
import os
import pickle
import scipy.sparse

def binary_mask_iou(m1, m2):
    mask_area1 = np.sum(m1)
    mask_area2 = np.sum(m2)
    intersection = np.logical_and(m1, m2)
    intersection_area = np.sum(intersection)
    union = mask_area1 + mask_area2 - intersection_area
    return intersection_area / union, intersection_area, union

def main():
    # num_exp = 10

    # all_masks = {}
    # avg_sparsity = []
    # for exp in range(10, num_exp+10):
    #     path = 'eval_checkpoints/memorize_%s/masks' % str(exp)
    #     # read files in path 
    #     files = os.listdir(path)
    #     masks = {}
    #     for file in files:
    #         with open(os.path.join(path, file), 'rb') as f:
    #             mask = pickle.load(f)
    #             # convert to int
    #             mask = mask.astype(int)
    #             masks[file.split('.pkl')[0]] = mask
    #     all_masks[exp] = masks
    #     avg_sparsity.append(np.mean([np.mean(mask) for mask in masks.values()]))
    # # take pairwise IOU
        
    num_exp = 10

    all_masks = {}
    avg_sparsity = []
    for exp in range(0, num_exp):
        path = 'eval_checkpoints_ap/memorize_%s/masks' % str(exp)
        # read files in path 
        files = os.listdir(path)
        masks = {}
        for file in files:
            with open(os.path.join(path, file), 'rb') as f:
                mask = pickle.load(f)
                mask = np.array(mask)
                # convert to int
                mask = mask.astype(int)
                masks[file.split('.json')[0]] = mask
        all_masks[exp] = masks
        avg_sparsity.append(np.mean([np.mean(mask) for mask in masks.values()]))
    # take pairwise IOU
        

    layer_wise_iou = {}
    # initialize layer wise iou 
    for key in all_masks[0].keys():
        layer_wise_iou[key] = []

    for exp in range(0,num_exp):
        for exp2 in range(exp+1, num_exp):
            iou = {}
            for key in all_masks[exp].keys():
                iou, intersection, union = binary_mask_iou(all_masks[exp][key], all_masks[exp2][key])
                print("IOU between exp %s and exp %s: " % (exp, exp2), iou, intersection, union)
                layer_wise_iou[key].append(iou)
    # print layer wise iou
    for key in layer_wise_iou.keys():
        print("Layer %s: " % key, np.mean(layer_wise_iou[key]))

    # plot sparsity
    print("Average sparsity: ", avg_sparsity)
    print("Average sparsity: ", np.mean(avg_sparsity))

    import matplotlib.pyplot as plt 
    plt.plot(avg_sparsity, marker='o')
    plt.xlabel('Experiment')
    plt.xticks(range(num_exp))
    plt.ylabel('Average Density for FFNs(%)')
    plt.title('Density of masks for every experiment')

    plt.savefig('sparsity_ap.png')

    # plot IOU for each layer as bar plot
    layer_names = list(layer_wise_iou.keys())
    layer_iou = [np.mean(layer_wise_iou[key]) for key in layer_names]

    plt.figure()
    plt.bar(layer_names, layer_iou)
    plt.xlabel('Layer')
    plt.ylabel('Average of Pairwise IOU')
    plt.title(f'Average of Pairwise IOU for {num_exp} experiments')
    plt.xticks(range(len(layer_names)), range(1, len(layer_names)+1))

    plt.savefig('layer_iou_ap.png')




if __name__ == '__main__':
    main()




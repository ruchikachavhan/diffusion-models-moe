import imageio
import os

res_path = os.path.join('results_1/stable-diffusion/fine-tuned-relu/', 'runwayml/stable-diffusion-v1-5')
adjective = 'white'

path = os.path.join(res_path, 'modularity', adjective)
timsteps = 51

files = [f'avg_neuron_value_timestep_{i}.png' for i in range(timsteps)]

ims = [imageio.imread(os.path.join(path, file)) for file in files]
imageio.mimwrite(os.path.join(path, 'movie.gif'), ims, duration=0.7)


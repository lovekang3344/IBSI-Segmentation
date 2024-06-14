import imageio
from glob import glob

path1_list = sorted(glob('./checkpoint/ConvTranspose2d_result/'+"*_28.png"))
duration = 2
loop = 0

pics_list = []
for image_name in path1_list:
    im = imageio.v3.imread(image_name)
    pics_list.append(im)
imageio.mimsave('ConvTranspose2d_28.gif', pics_list, 'GIF', fps=0.8, loop=loop)

path2_list = sorted(glob('./checkpoint/Upsampling_result/'+"*_28.png"))
pics_list = []
for image_name in path2_list:
    im = imageio.v3.imread(image_name)
    pics_list.append(im)
imageio.mimsave('Upsampling_28.gif', pics_list, 'GIF', fps=0.8, loop=loop)

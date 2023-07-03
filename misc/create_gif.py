import imageio
import numpy as np
import cv2
import os

def create_gif_hstack(source_dir, duration, output_name='gif.gif'):
    filenames = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    filenames_latent = [x for x in filenames if 'latent' in x]
    filenames_img = [x for x in filenames if 'latent' not in x]

    images = []
    for name_img, name_latent in zip(filenames_img, filenames_latent):
        img = cv2.imread(os.path.join(source_dir, name_img))
        latent = cv2.imread(os.path.join(source_dir, name_latent))

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        latent = cv2.cvtColor(latent, cv2.COLOR_RGBA2RGB) # cv2.cvtColor(latent, cv2.COLOR_BGR2GRAY)
        latent = cv2.resize(latent, (256, 256))

        image = np.hstack((img, latent))
        images.append(image)

    imageio.mimsave(os.path.join(source_dir, output_name), images, duration=duration)

def create_gif(source_dir, duration, output_name='gif.gif'):
    filenames = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    images = []
    for filename in filenames:
        image = imageio.imread(os.path.join(source_dir, filename))
        #image = np.resize(image, (256, 256))
        images.append(image)

    imageio.mimsave(os.path.join(source_dir, output_name), images, duration=duration)

if __name__ == '__main__':

    dataset_name = 'Pelvis_2.1_repo_no_mask'
    analysis_name = 'github-analysis'
    reports_dir = './reports/'

    #create_gif_hstack(os.path.join(reports_dir, dataset_name, analysis_name, 'all'), duration=0.5, output_name='img.gif')
    create_gif(os.path.join(reports_dir, dataset_name, analysis_name, 'img'), duration=0.5, output_name='img.gif')
    create_gif(os.path.join(reports_dir, dataset_name, analysis_name, 'latent'), duration=0.5, output_name='latent.gif')

    print('May be the force with you.')

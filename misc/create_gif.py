import imageio
import os

def create_gif(source_dir, duration, output_name='gif.gif'):
    filenames = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    images = []
    for filename in filenames:
        images.append(imageio.imread(os.path.join(source_dir, filename)))
    imageio.mimsave(os.path.join(source_dir, output_name), images, duration=duration)

if __name__ == '__main__':

    dataset_name = 'Pelvis_2.1_repo_no_mask'
    analysis_name = 'github-analysis'
    reports_dir = './reports/'

    create_gif(os.path.join(reports_dir, dataset_name, analysis_name), duration=0.5, output_name='img.gif')

    print('May be the force with you.')

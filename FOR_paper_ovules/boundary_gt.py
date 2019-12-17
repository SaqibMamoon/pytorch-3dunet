import h5py

from augment.transforms import StandardLabelToBoundary

in_file = '/home/adrian/workspace/ilastik-datasets/Ovules/val/N_464_ds2x.h5'
out_file = '/home/adrian/workspace/ilastik-datasets/Ovules/val/N_464_ds2x_boundary.h5'

# in_file = '/home/adrian/workspace/ilastik-datasets/MuviSPIM/GT/t00045_s00_uint8_gt_cropped_final.h5'
# out_file = '/home/adrian/workspace/ilastik-datasets/MuviSPIM/GT/t00045_s00_uint8_gt_cropped_final_boundary.h5'


with h5py.File(in_file, 'r') as f:
    label = f['label'][...]

sigmas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

with h5py.File(out_file, 'w') as f:
    t = StandardLabelToBoundary(blur=False)
    f.create_dataset('boundary', data=t(label).astype('uint8'), compression='gzip')
    # for i, sigma in enumerate(sigmas):
    #     print(f'Processing sigma: {sigma}...')
    #     aug = StandardLabelToBoundary(blur=True, sigma=sigma)
    #     boundary = aug(label)
    #     f.create_dataset(f'boundary{i}', data=boundary.astype('uint8'), compression='gzip')

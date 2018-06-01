"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import numpy as np
from nilearn import image
from nilearn._utils.niimg_conversions import check_niimg


def dice(mask_file1, mask_file2):
    mask_data1 = check_niimg(mask_file1).get_data() > 0
    mask_data2 = check_niimg(mask_file2).get_data() > 0
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)

if __name__ == '__main__':
    template_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes/Salma',
                                'inhouse_mouse_perf/template/with_mask')
    registered_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes',
                                  'Salma/inhouse_mouse_perf_preprocessed_mine',
                                  'mouse_191851/reoriented')

    template_tissues_imgs = [
        os.path.join(template_dir, 'c{0}head100.nii'.format(n))
        for n in range(1, 4)]
    template_mask_img = image.math_img(
        'np.max(imgs, axis=-1) > .000001', imgs=template_tissues_imgs)
    template_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=template_mask_img,
        imgs=template_tissues_imgs)

    registered_tissues_imgs = [
        os.path.join(
            registered_dir,
            'c{0}anat_n0_unifized_affine_general_warped_clean_hd.nii'.format(n))
        for n in range(1, 4)]
    registered_mask_img = image.math_img(
        'np.max(imgs, axis=-1) > .000001', imgs=registered_tissues_imgs)
    registered_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=registered_mask_img,
        imgs=registered_tissues_imgs)

    for label in [1, 2, 3]:
        tissue_template_img = image.math_img('img=={0}'.format(label),
                                             img=template_img)
        tissue_registered_img = image.math_img('img=={0}'.format(label),
                                               img=registered_img)
        print(dice(tissue_template_img, tissue_registered_img))

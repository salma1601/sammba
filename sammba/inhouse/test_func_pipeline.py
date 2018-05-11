"""
Tests comparing Nad's outputs to sammba's
"""
import os
import numpy as np
import nibabel
from sklearn.datasets.base import Bunch
from sammba.registration import TemplateRegistrator


def load_inhouse_mouse():
    # No weight file is applied so comparison must stop at affine level of anatomical
    params = {}
    original_dir = '/home/bougacha/inhouse_mouse'
    params['anat'] = os.path.join(original_dir,
                        'MINDt-AmylNet-64803__20170405__194252__T1_FLASH_3D_5min_GOP__dirnum__7.nii.gz')
    params['func'] = os.path.join(original_dir,
                        'MINDt-AmylNet-64803__20170405__195120__GE_EPI_sat_Grandjean_GOP__dirnum__8.nii.gz')
    params['anat'] = os.path.join(original_dir, 'anat_n0.nii.gz')
    params['func'] = os.path.join(original_dir, 'rs_n0.nii.gz')
    nad_dir = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/AmylNet/analysis20170725/MRIsessions/20170405_192109_MINDt_AmylNet_64803_1_1'
    params['unifized_anat'] = os.path.join(nad_dir, 'anat_n0_Un.nii.gz')
    params['unifized_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBe.nii.gz')
    params['allineated_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBeAl.nii.gz')
    params['allineated_anat'] = os.path.join(nad_dir, 'anat_n0_UnAa.nii.gz')
    params['normalization_pretransform'] = os.path.join(nad_dir,
                                              'anat_n0_UnBmBeAl.aff12.1D')
    params['warped_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBeAl.nii.gz')
    params['coregistered_anat'] = os.path.join(nad_dir,
                                     'anat_n0_Un_Op_rs_n0_TsAvAvN3.nii.gz')
    params['func_to_anat_transform'] = os.path.join(
        nad_dir, 'anat_n0_Un_Op_rs_n0_TsAvAvN3.aff12.1D')
    params['mean_func'] = os.path.join(nad_dir, 'rs_n0_TsAvAv.nii.gz')
    params['mean_unbiased_func'] = os.path.join(nad_dir, 'rs_n0_TsAvAvN3.nii.gz')
    params['undistorded_func'] = os.path.join(nad_dir, 'rs_n0_TsAv_NaMe.nii.gz')
    params['normalized_func'] = os.path.join(nad_dir, 'rs_n0_TsAv_NaMeNa.nii.gz')
    params['normalized_anat'] = os.path.join(nad_dir, 'T1anat_n0_UnAaQw.nii.gz')
    params['template'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/head100.nii.gz'
    params['template_brain'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/brain100.nii.gz'
    params['dilated_template_mask'] = None
    params['conv'] = .005

    return Bunch(**params)

def load_inhouse_lemurs():
#    '20171025_192656_MD1704_Mc288BC_P01_1_1' : failed brain extraction (left eye included)

    params = {}
    nad_dir = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouselemur/MLOct2017/analysis20171030/20171025_173010_MD1704_Mc943GKBC_P01_1_1'
    salma_dir = '/home/bougacha/inhouse_lemurs/20171025_173010_MD1704_Mc943GKBC_P01_1_1'
    params['anat'] = os.path.join(salma_dir, 'anat_n0.nii.gz')
    params['func'] = os.path.join(salma_dir, 'rs_n0.nii.gz')
    params['unifized_anat'] = os.path.join(nad_dir, 'anat_n0_Un.nii.gz')
    params['unifized_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBe.nii.gz')
    params['allineated_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBeAl.nii.gz')
    params['allineated_anat'] = os.path.join(nad_dir, 'anat_n0_UnAa.nii.gz')
    params['normalization_pretransform'] = os.path.join(nad_dir,
                                              'anat_n0_UnBmBeAl.aff12.1D')
    params['warped_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBeAl.nii.gz')
    params['normalized_anat'] = os.path.join(nad_dir, 'anat_n0_UnAaQw.nii.gz')
    params['anat_in_func_space'] = os.path.join(nad_dir,
                                     'anat_n0_Un_Op_rs_n0_TsAvAvN3.nii.gz')
    params['func_to_anat_transform'] = os.path.join(
        nad_dir, 'anat_n0_Un_Op_rs_n0_TsAvAvN3.aff12.1D')
    params['mean_func'] = os.path.join(nad_dir, 'rs_n0_TsAvAv.nii.gz')
    params['mean_unbiased_func'] = os.path.join(nad_dir, 'rs_n0_TsAvAvN3.nii.gz')
    params['undistorded_func'] = os.path.join(nad_dir, 'rs_n0_TsAv_NaMe.nii.gz')
    params['func_perslice_warp0'] = os.path.join(nad_dir, 'rs_n0_TsAvAvN3_slice_0001_Na_WARP.nii.gz')
    params['normalized_func'] = os.path.join(nad_dir, 'rs_n0_TsAv_NaMeNa.nii.gz')
    params['template'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/head100.nii.gz'
    params['template_brain'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouselemur/RsLemurs/analysis20171017/Qw4_meanhead_brain_Na_200.nii.gz'
    params['dilated_template_mask'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouselemur/RsLemurs/analysis20171017/aff3_unionmaskdil5_Na_200.nii.gz'
    params['conv'] = .01

    return Bunch(**params)


def check_same_image(file1, file2, decimal=None):
    img1 = nibabel.load(file1)
    img2 = nibabel.load(file2)
    try:
        if decimal is None:
            np.testing.assert_array_equal(img1.affine, img2.affine)
        else:
            np.testing.assert_array_almost_equal(img1.affine, img2.affine,
                                                 decimal=decimal)
    except AssertionError:
        diff = np.linalg.norm(img1.affine - img2.affine)
        raise AssertionError('images do not have the same affine: norm'
                             'difference is {}'.format(diff))
    try:
        np.testing.assert_array_equal(img1.get_data(), img2.get_data())
    except AssertionError:
        diff = np.linalg.norm(img1.get_data() - img2.get_data())
        raise AssertionError('images do not have the same data: norm'
                             'difference is {}'.format(diff))


def test_template_registrator_mouse():
    data = load_inhouse_mouse()
    registrator = template_registrator.TemplateRegistrator(brain_volume=400, caching=True,
          dilated_template_mask=data.dilated_template_mask,
          mask_clipping_fraction=0.2,
          output_dir='/home/bougacha/inhouse_mouse_preprocessed2',
          template=data.template,
          template_brain_mask='/home/bougacha/inhouse_mouse/brain100_binarized.nii.gz',
          use_rats_tool=True, verbose=True, convergence=data.conv)

    # First check the brain extraction parameters are the same
    unifized_anat, brain = registrator.segment(data.anat)
    # Rats tool changede the affine ?
    check_same_image(unifized_anat, data.unifized_anat)
    check_same_image(brain, data.unifized_anat_brain)

    # Only check affine normalization
    registrator.fit_anat(data.anat)
    check_same_image(registrator.registered_anat.replace('_warped', ''),
                     data.allineated_anat)

    registrator = coregistrator.AnatCoregistrator(brain_volume=400, caching=True,
          output_dir='/home/bougacha/inhouse_mouse_preprocessed2',
          use_rats_tool=True, verbose=True)

    registrator.fit_anat(data.anat)
    check_same_image(registrator.anat_brain_, data.unifized_anat_brain)

    registrator.fit_modality(data.func, 'func', t_r=1.)
    check_same_image(registrator.undistorted_func.replace('_perslice_oblique', '_tstat'),
                     data.mean_func) # Ok visually
    check_same_image(registrator.undistorted_func.replace('_perslice_oblique', ''),
                     data.mean_unbiased_func)
    check_same_image(registrator.undistorted_func.replace('_perslice_oblique',
                                                          '_sl9_oblique_resampled_qwarped_WARP'),
                     data.func_perslice_warp0)

    check_same_image(registrator.anat_in_func_space, data.anat_in_func_space)
    check_same_image(registrator.undistorted_func, data.undistorded_func)

    # comparison of nmis: the lower nmi the better alignment
    assert_less_equal(np.sum(img_s.get_data() > img_nad.get_data()),
                      np.sum(img_s.get_data() < img_nad.get_data()))
    assert_less_equal(img_s.get_data().mean(), img_nad.get_data().mean())

def test_template_registrator():
    data = load_inhouse_lemurs()
    registrator = TemplateRegistrator(brain_volume=1850, caching=True,
          dilated_template_mask=data.dilated_template_mask,
          mask_clipping_fraction=0.2,
          output_dir='/home/bougacha/inhouse_lemurs_preprocessed/20171025_173010_MD1704_Mc943GKBC_P01_1_1',
          template=data.template,
          template_brain_mask='/home/bougacha/inhouse_lemurs/Qw4_meanhead_brain_Na_200_binarized.nii.gz',
          use_rats_tool=True, verbose=True)

    # First check the brain extraction parameters are the same
    unifized_anat, brain_mask = registrator.segment(data.anat)
    check_same_image(unifized_anat, data.unifized_anat)
    check_same_image(brain_mask, data.unifized_anat_brain)

    registrator.fit_anat(data.anat)
    check_same_image(registrator.registered_anat.replace('_warped', ''),
                     data.allineated_anat)
    check_same_image(registrator.registered_anat, data.normalized_anat)

    registrator.fit_modality(data.anat, voxel_size=(.3, .3, .3))
    check_same_image(registrator.anat_in_func_space, data.anat_in_func_space)
    check_same_image(registrator.undistorted_func, data.undistorded_func)
    
    registrator = Coregistrator(brain_volume=1850, caching=True,
          output_dir='/home/bougacha/inhouse_lemurs_preprocessed/20171025_192656_MD1704_Mc288BC_P01_1_1',
          use_rats_tool=True, verbose=True)

    registrator.fit_anat(data.anat)
    check_same_image(registrator.normalized_anat, data.normalized_anat)
    registrator.fit_modality(data.anat, voxel_size=(.3, .3, .3))
    check_same_image(registrator.anat_in_func_space, data.anat_in_func_space)
    check_same_image(registrator.undistorted_func, data.undistorded_func)

    registrator = TemplateRegistrator(brain_volume=400, caching=True,
      dilated_template_mask=data.dilated_template_mask,
      mask_clipping_fraction=0.2,
      output_dir='/home/bougacha/inhouse_lemurs_preprocessed/20171025_192656_MD1704_Mc288BC_P01_1_1',
      template='/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/head100.nii.gz',
      template_brain_mask='/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/mask100.nii.gz',
      use_rats_tool=True, verbose=True)
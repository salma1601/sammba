"""
Tests comparing Nad's outputs to sammba's
"""
import os
import glob
from nose.tools import assert_less_equal
import numpy as np
import nibabel
from nilearn import image
from sklearn.datasets.base import Bunch
from sammba.registration import template_registrator
from sammba.registration.perfusion import coregister as coregister_perf
from sammba.modality_processors import perf_fair_niiptbl_proc
from nilearn._utils.niimg_conversions import check_niimg


def load_inhouse_mouse_perf():
    # No weight file is applied so comparison must stop at affine level of anatomical
    params = {}
    original_dir = '/home/bougacha/inhouse_mouse_perf/mouse_191851'
    params['anat'] = os.path.join(original_dir, 'anat_n0.nii.gz')
    params['perf'] = os.path.join(original_dir, 'perfFAIREPI_n2.nii.gz')
    params['perf_proc'] = os.path.join(original_dir, 'perfFAIREPI_n2_proc.nii.gz')
    params['m0'] = os.path.join(original_dir, 'perfFAIREPI_n2_M0.nii.gz')
    params['unbiased_m0'] = os.path.join(original_dir, 'perfFAIREPI_n2_M0_N3.nii.gz')
    # _calc: don't know what it is
    nad_dir = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/BECIM/MRI-11.7T/analysis20170905/MRIsessions/20170104_191851_MINDt_ROMANE_56635_1_1'
    params['unifized_anat'] = os.path.join(nad_dir, 'anat_n0_Un.nii.gz')
    params['unifized_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBe.nii.gz')
    params['allineated_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBeAl.nii.gz')
    params['allineated_anat'] = os.path.join(nad_dir, 'anat_n0_UnAa.nii.gz')
    params['normalization_pretransform'] = os.path.join(nad_dir,
                                              'anat_n0_UnBmBeAl.aff12.1D')
    params['warped_anat_brain'] = os.path.join(nad_dir, 'anat_n0_UnBmBeAl.nii.gz')
    params['anat_in_perf_space'] = os.path.join(nad_dir,
                                     'anat_n0_Un_Op_perfFAIREPI_n2_M0_N3.nii.gz')
    params['anat_to_perf_transform'] = os.path.join(
        nad_dir, 'anat_n0_Un_Op_perfFAIREPI_n2_M0_N3.aff12.1D')
    params['unbiased_m0'] = os.path.join(nad_dir, 'perfFAIREPI_n2_M0_N3.nii.gz')
    params['undistorded_perf'] = os.path.join(nad_dir, 'perfFAIREPI_n2_M0_N3_NaMe.nii.gz')
    params['perf_undistort_warps'] = sorted(glob.glob('perfFAIREPI_n2_M0_N3_slice_*_Na_WARP.nii.gz'))
    params['normalized_perf'] = os.path.join(nad_dir, 'perfFAIREPI_n2_M0_N3_NaMeNa.nii.gz')
    params['normalized_anat'] = os.path.join(nad_dir, 'anat_n0_UnAaQw.nii.gz')
    params['template'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/head100.nii.gz'
    params['template_brain'] = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/brain100.nii.gz'
    params['dilated_template_mask'] = None
    params['normalized_cbf'] = '/home/bougacha/test_design/cbfs/cbf2_1.nii.gz'
    params['conv'] = .005

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


def dice(mask_file1, mask_file2):
    mask_data1 = check_niimg(mask_file1).get_data() > 0
    mask_data2 = check_niimg(mask_file2).get_data() > 0
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)


from skimage import measure
from sammba.externals.nipype.utils.filemanip import fname_presuffix


def compute_mask_contour(mask_file, write_dir=None, out_file=None):
    mask_img = nibabel.load(mask_file)
    vertices, _ = measure.marching_cubes(mask_img.get_data(), 0)  #marching_cubes_lewiner
    vertices_minus = np.floor(vertices).astype(int)
    vertices_plus = np.ceil(vertices).astype(int)
    contour_data = np.zeros(mask_img.shape)
    contour_data[vertices_minus.T[0],
                 vertices_minus.T[1],
                 vertices_minus.T[2]] = 1
    contour_data[vertices_plus.T[0],
                 vertices_plus.T[1],
                 vertices_plus.T[2]] = 1
    contour_img = image.new_img_like(mask_img, contour_data)
    if write_dir is None:
        write_dir = os.getcwd()

    if out_file is None:
        out_file = fname_presuffix(mask_file, suffix='_countour',
                                   newpath=write_dir)
    contour_img.to_filename(out_file)
    return out_file


def test_template_registrator_mouse():
    ###########################################################################
    # With default pipeline, test better coregistration/normalization than Nad
    ###########################################################################
    data = load_inhouse_mouse_perf()
    registrator = template_registrator.TemplateRegistrator(
        brain_volume=400,
        caching=True,
        dilated_template_mask=data.dilated_template_mask,
        output_dir='/home/bougacha/inhouse_mouse_perf_preprocessed_mine/mouse_191851',
        template=data.template,
        template_brain_mask='/home/bougacha/inhouse_mouse/brain100_binarized.nii.gz',
        use_rats_tool=True, verbose=True, convergence=data.conv,
        registration_kind='nonlinear')
    registrator.fit_anat(data.anat)
    brain_mask_file = registrator.anat_brain_.replace(
        '.nii', '_for_extraction_brain_mask.nii')
    normalized_brain = registrator.fit_transform_anat_like(
        data.anat, registrator.anat_brain_, brain_mask_file, 
        interpolation='nearestneighbour')

    _, registered_anat_brain = registrator.segment(registrator.registered_anat,
                                                   unifize=False)
    print(dice(registered_anat_brain, registrator.template_brain_))

    _, registered_anat_brain = registrator.segment(registrator.registered_anat,
                                                   unifize=False)

    _, registered_m0_brain = registrator.segment(registrator.registered_perf_,
                                                 unifize=False)
    registered_m0_brain_mask_file = base.compute_brain_mask(
        registrator.registered_perf_, 200, registrator.output_dir,
        bias_correct=False, use_rats_tool=False, upper_cutoff=.9,
        lower_cutoff=011,
        opening=5, closing=1, dilation_size=(0, 0, 0))
    rats_registered_m0_brain_mask_file = base.compute_brain_mask(
        registrator.registered_perf_, 150,
        registrator.output_dir + '/rats_perf',
        bias_correct=False, use_rats_tool=True)

    from nilearn import image
    registered_anat_brain = registrator.template_brain_mask

    anat_brain_contour_file = compute_mask_contour(
        registered_anat_brain, write_dir=registrator.output_dir)

    template_brain_contour_file = compute_mask_contour(
        registrator.template_brain_mask, write_dir=registrator.output_dir)

    # Zero out the brain template outside the perfusion FOV
    m0_mask_img = image.math_img('img>10', img=registrator.registered_perf_)
    cut_template_brain_img = image.math_img(
        'img1 * img2',
        img1=registrator.template_brain_mask,
        img2=m0_mask_img)
    cut_template_brain_file = fname_presuffix(registrator.template_brain_mask,
                                              suffix='_cut',
                                              newpath=registrator.output_dir)
    cut_template_brain_img.to_filename(cut_template_brain_file)
    cut_template_brain_contour_file = compute_mask_contour(
        cut_template_brain_file, write_dir=registrator.output_dir)


    # Zero out the brain template outside the perfusion FOV
    cleaned_template_brain_contour_img = image.math_img(
        'img1 * img2',
        img1=template_brain_contour_file,
        img2=m0_mask_img)
    perf_artificial_contour_img = image.math_img(
        'img1!=img2',
        img1=cut_template_brain_contour_file,
        img2=cleaned_template_brain_contour_img)

    # Compute contour
    m0_brain_mask_img = image.math_img('img>0',
                                       img=rats_registered_m0_brain_mask_file)
    m0_brain_mask_file = fname_presuffix(registrator.registered_perf_,
                                         suffix='_brain_mask',
                                         newpath=registrator.output_dir)
    m0_brain_mask_img.to_filename(m0_brain_mask_file)
    m0_brain_contour_file = compute_mask_contour(
        m0_brain_mask_file, write_dir=registrator.output_dir)
    m0_clean_contour_img = image.math_img('(img1 * img3 - img2) >0',
                                          img1=m0_brain_contour_file,
                                          img2=perf_artificial_contour_img,
                                          img3=m0_mask_img)
    m0_clean_contour_img.to_filename(
        fname_presuffix(m0_brain_contour_file,
                        suffix='_clean',
                        newpath=registrator.output_dir))
    print(dice(m0_clean_contour_img, cleaned_template_brain_contour_img))

    # anat normalization
    l = utils.LocalBistat()
    l.inputs.neighborhood = ('SPHERE', .12)
    l.inputs.in_files = [registrator.registered_anat, registrator.template]
    l.inputs.stat = ['spearman', 'normuti']
    l_out = l.run()
    spearman_img = image.index_img(l_out.outputs.out_file, 0)
    nmi_img = image.index_img(l_out.outputs.out_file, 1)

    l.inputs.in_files = [data.normalized_anat, registrator.template]
    l_out = l.run()
    nad_spearman_img = image.index_img(l_out.outputs.out_file, 0)
    nad_nmi_img = image.index_img(l_out.outputs.out_file, 1)

    assert_less_equal(np.sum(nmi_img.get_data() > nad_nmi_img.get_data()),
                      np.sum(nmi_img.get_data() < nad_nmi_img.get_data()))
    assert_less_equal(nmi_img.get_data().mean(), nad_nmi_img.get_data().mean())

    assert_less_equal(np.sum(spearman_img.get_data() < nad_spearman_img.get_data()),
                      np.sum(spearman_img.get_data() > nad_spearman_img.get_data()))
    assert_less_equal(nad_spearman_img.get_data().mean(),
                      spearman_img.get_data().mean())


    registrator.fit_modality(data.m0, 'perf')
    # M0 coregistration
    l.inputs.in_files = [registrator.anat_in_perf_space, data.m0]
    l.inputs.stat = ['spearman', 'normuti']
    l_out = l.run()
    spearman_img = image.index_img(l_out.outputs.out_file, 0)
    nmi_img = image.index_img(l_out.outputs.out_file, 1)

    l.inputs.in_files = [data.anat_in_perf_space, data.m0]
    l_out = l.run()
    nad_spearman_img = image.index_img(l_out.outputs.out_file, 0)
    nad_nmi_img = image.index_img(l_out.outputs.out_file, 1)
    
    assert_less_equal(np.sum(nmi_img.get_data() > nad_nmi_img.get_data()),
                      np.sum(nmi_img.get_data() < nad_nmi_img.get_data()))
    assert_less_equal(nmi_img.get_data().mean(), nad_nmi_img.get_data().mean())

    assert_less_equal(np.sum(spearman_img.get_data() < nad_spearman_img.get_data()),
                      np.sum(spearman_img.get_data() > nad_spearman_img.get_data()))
    assert_less_equal(nad_spearman_img.get_data().mean(),
                      spearman_img.get_data().mean())


    ###########################################################################
    # With stratiforward pipeline, test normalization Stats are good
    ###########################################################################
    # M0 normalization
    l.inputs.in_files = [registrator.registered_perf_, registrator.template]
    l.inputs.stat = ['spearman', 'normuti']
    l_out = l.run()
    spearman_img = image.index_img(l_out.outputs.out_file, 0)
    nmi_img = image.index_img(l_out.outputs.out_file, 1)
   
    # XXX: no matching at all, repeat with CBF
    assert_less_equal(np.sum(nmi_img.get_data() > nad_nmi_img.get_data()),
                      np.sum(nmi_img.get_data() < nad_nmi_img.get_data()))
    assert_less_equal(nmi_img.get_data().mean(), nad_nmi_img.get_data().mean())

    # Check normalization to CBF
    if not os.path.isfile(data.perf_proc):
        perf = data.perf
        perf_fair_niiptbl_proc(perf, 2800.)

    cbf_file = '/home/bougacha/inhouse_mouse_perf_preprocessed_mine/mouse_191851/perfFAIREPI_n2_cbf.nii.gz'
    rcbf_file = '/home/bougacha/inhouse_mouse_perf_preprocessed_mine/mouse_191851/perfFAIREPI_n2_rcbf.nii.gz'
    if not os.path.isfile(cbf_file):
        cbf = image.index_img(data.perf_proc, 13)
        cbf.to_filename(cbf_file)
    if not os.path.isfile(rcbf_file):
        rcbf = image.index_img(data.perf_proc, 12)
        rcbf.to_filename(rcbf_file)

    normalized_cbf = registrator.transform_modality_like(cbf_file, 'perf')
    normalized_rcbf = registrator.transform_modality(rcbf_file, 'perf')

    registrator = coregistrator.AnatCoregistrator(brain_volume=400, caching=True,
          output_dir='/home/bougacha/inhouse_mouse_perf_preprocessed',
          use_rats_tool=True, verbose=True)

    # comparison of nmis: the lower nmi the better alignment
    assert_less_equal(np.sum(img_s.get_data() > img_nad.get_data()),
                      np.sum(img_s.get_data() < img_nad.get_data()))
    assert_less_equal(img_s.get_data().mean(), img_nad.get_data().mean())

    ###########################################################################
    # With Nad inputs, test exactly same outputs
    ###########################################################################
    # First check the brain extraction parameters are the same
    unifized_anat, brain = registrator.segment(data.anat)
    check_same_image(brain, data.unifized_anat_brain)

    # Only check affine normalization
    registrator = template_registrator.TemplateRegistrator(
        brain_volume=400, caching=True,
        dilated_template_mask=data.dilated_template_mask,
        output_dir='/home/bougacha/inhouse_mouse_perf_preprocessed_as_nad/mouse_191851',
        template=data.template,
        template_brain_mask='/home/bougacha/inhouse_mouse/brain100_binarized.nii.gz',
        use_rats_tool=True, verbose=True, convergence=data.conv,
        registration_kind='affine')
    registrator.fit_anat(data.anat)
    check_same_image(registrator._unifized_anat_, data.unifized_anat)
    check_same_image(registrator.anat_brain_, data.unifized_anat_brain)
    check_same_image(registrator.registered_anat, data.allineated_anat)

    # Check M0 coregistration
    # N3 is used, so must be done manually
    coregistration = coregister_perf(
        registrator._unifized_anat_, data.unbiased_m0,
        registrator.output_dir,
        anat_brain_file=registrator.anat_brain_,
        m0_brain_file=None,
        prior_rigid_body_registration=False,
        caching=registrator.caching)
    registrator.undistorted_perf = coregistration.coreg_m0_
    registrator._perf_undistort_warps = coregistration.coreg_warps_
    registrator.anat_in_perf_space = coregistration.coreg_anat_
    registrator._anat_to_perf_transform = coregistration.coreg_transform_

    for (warp_file, nad_warp_file) in zip(registrator._perf_undistort_warps,
                                          data.perf_undistort_warps):
        check_same_image(warp_file, nad_warp_file)

    check_same_image(registrator.anat_in_perf_space, data.anat_in_perf_space)
    check_same_image(registrator.undistorted_perf, data.undistorded_perf)
    np.testing.assert_array_equal(
        np.loadtxt(registrator._anat_to_perf_transform),
        np.loadtxt(data.anat_to_perf_transform))
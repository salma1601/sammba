import os
import glob
import numpy as np
import nibabel
from nilearn import image
from sammba.registration import template_registrator


mice_perfs = glob.glob('/home/Promane/2014-ROMANE/' +
    '5_Experimental-Plan-Experiments-Results/mouse/BECIM/MRI-11.7T/' +
    'analysis20170905/MRIsessions/2017*/perfFAIREPI_n0.nii.gz')
mice_perfs = mice_perfs[:10]
mice_dirs = [os.path.dirname(p) for p in mice_perfs]
anat_files = [os.path.join(d, 'anat_n0.nii.gz') for d in mice_dirs]
m0_files = [os.path.join(d, 'perfFAIREPI_n0_M0.nii.gz') for d in mice_dirs]
perf_proc_files = [os.path.join(d, 'perfFAIREPI_n0_proc.nii.gz')
                   for d in mice_dirs]

template_file = os.path.join(
    '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse',
    'MRIatlases/MIC40C57/20170223/head100.nii.gz')
template_brain_mask_file = os.path.join('/home/bougacha/inhouse_mouse',
                                        'brain100_binarized.nii.gz')
spm_data_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes/Salma/',
                            'inhouse_mouse_perf_preprocessed_mine')
for mouse_dir, anat_file, m0_file, perf_proc_file in zip(mice_dirs,
                                                         anat_files,
                                                         m0_files,
                                                         perf_proc_files)[1:2]:
    mouse_id = os.path.basename(mouse_dir)
    mouse_id = mouse_id.replace('20170104', 'mouse')
    mouse_id = mouse_id.replace('_MINDt_ROMANE_56634_1_1', '')
    output_dir = os.path.join(
        '/home/bougacha/inhouse_mouse_perf_preprocessed_mine', mouse_id)
    registrator = template_registrator.TemplateRegistrator(
        brain_volume=400,
        dilated_template_mask=None,
        output_dir=output_dir,
        template=template_file,
        template_brain_mask=template_brain_mask_file)

    registrator.fit_anat(anat_file)
    # Save uncompressed image to output spm directory
    target_dir = os.path.join(spm_data_dir, mouse_id)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    target_anat_base, _ = os.path.splitext(
        os.path.basename(registrator.registered_anat))
    registered_anat_img = nibabel.load(registrator.registered_anat)
    registered_anat_img.to_filename(os.path.join(target_dir, target_anat_base))

    registrator.fit_modality(m0_file, 'perf')
    cbf_file = os.path.join(output_dir, 'perfFAIREPI_n0_cbf.nii.gz')
    cbf = image.index_img(perf_proc_file, 13)
    cbf.to_filename(cbf_file)

    normalized_cbf = registrator.transform_modality_like(cbf_file, 'perf')

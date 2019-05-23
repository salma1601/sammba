# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..dti import XFibres5


def test_XFibres5_inputs():
    input_map = dict(
        all_ard=dict(
            argstr='--allard',
            xor=('no_ard', 'all_ard'),
        ),
        args=dict(argstr='%s', ),
        burn_in=dict(
            argstr='--burnin=%d',
            usedefault=True,
        ),
        burn_in_no_ard=dict(
            argstr='--burnin_noard=%d',
            usedefault=True,
        ),
        bvals=dict(
            argstr='--bvals=%s',
            mandatory=True,
        ),
        bvecs=dict(
            argstr='--bvecs=%s',
            mandatory=True,
        ),
        cnlinear=dict(
            argstr='--cnonlinear',
            xor=('no_spat', 'non_linear', 'cnlinear'),
        ),
        dwi=dict(
            argstr='--data=%s',
            mandatory=True,
        ),
        environ=dict(
            nohash=True,
            usedefault=True,
        ),
        f0_ard=dict(
            argstr='--f0 --ardf0',
            xor=['f0_noard', 'f0_ard', 'all_ard'],
        ),
        f0_noard=dict(
            argstr='--f0',
            xor=['f0_noard', 'f0_ard'],
        ),
        force_dir=dict(
            argstr='--forcedir',
            usedefault=True,
        ),
        fudge=dict(argstr='--fudge=%d', ),
        gradnonlin=dict(argstr='--gradnonlin=%s', ),
        logdir=dict(
            argstr='--logdir=%s',
            usedefault=True,
        ),
        mask=dict(
            argstr='--mask=%s',
            mandatory=True,
        ),
        model=dict(argstr='--model=%d', ),
        n_fibres=dict(
            argstr='--nfibres=%d',
            mandatory=True,
            usedefault=True,
        ),
        n_jumps=dict(
            argstr='--njumps=%d',
            usedefault=True,
        ),
        no_ard=dict(
            argstr='--noard',
            xor=('no_ard', 'all_ard'),
        ),
        no_spat=dict(
            argstr='--nospat',
            xor=('no_spat', 'non_linear', 'cnlinear'),
        ),
        non_linear=dict(
            argstr='--nonlinear',
            xor=('no_spat', 'non_linear', 'cnlinear'),
        ),
        output_type=dict(),
        rician=dict(argstr='--rician', ),
        sample_every=dict(
            argstr='--sampleevery=%d',
            usedefault=True,
        ),
        seed=dict(argstr='--seed=%d', ),
        update_proposal_every=dict(
            argstr='--updateproposalevery=%d',
            usedefault=True,
        ),
    )
    inputs = XFibres5.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value
def test_XFibres5_outputs():
    output_map = dict(
        dyads=dict(),
        fsamples=dict(),
        mean_S0samples=dict(),
        mean_dsamples=dict(),
        mean_fsamples=dict(),
        mean_tausamples=dict(),
        phsamples=dict(),
        thsamples=dict(),
    )
    outputs = XFibres5.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value

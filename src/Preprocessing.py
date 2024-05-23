import numpy as np
import snirf
from mne_nirs.visualisation import (
    plot_timechannel_quality_metric,
    plot_glm_group_topo)
from mne_nirs.preprocessing import (
    peak_power,
    scalp_coupling_index_windowed)
import mne
import mne_bids.stats
from mne_nirs import (
    channels,
    experimental_design,
    datasets,
    statistics,
    signal_enhancement)
from mne.preprocessing.nirs import (
    optical_density,
    temporal_derivative_distribution_repair)


def dpf_scholkman(age: float = 0.583,
                  wavelength: int = 760) -> float:
    alpha = 223.3
    term1 = alpha + (0.05624 * (age ** 0.8493))
    term2 = 5.723 * 10e-7 * (wavelength**3)
    term3 = (0.001245 * (wavelength ** 2)) - (0.9025 * wavelength)
    dpf = term1 - term2 + term3
    return dpf


def dpf_duncan(age: float = 0.583,
               wavelength: int = 690) -> float:
    valid_wavs = [690, 744, 807, 832]
    if not wavelength in valid_wavs:
        raise ValueError(
            "Duncan method only available for wavelengths " + repr(valid_wavs))
    dpf = {
        "690": 5.38 + 0.049 * age**0.877,
        "744": 5.11 + 0.106 * age**0.723,
        "807": 4.99 + 0.067 * age**0.814,
        "832": 4.67 + 0.062 * age**0.819,
    }
    return dpf[str(wavelength)]


def dpf(age: float = 0.583,
        wavelengths: tuple[int] = (760, 850),
        method: str = "scholkman") -> float:
    """
    Calculate Differential Path Factor (DPF).

    Parameters
    ----------

    age : float
        Age of the participant in years.
    wavelengths : tuple[int]
        Wavelengths.
    method : str
        Method for calculating DPF: "scholkman" (Default) or "duncan".

    Returns
    -------

    tupple
        DPF for each wavelength

    Notes
    -----

    Both the Scholkman and the Duncan methods involve least-squared estimates of the parameters involved. The Scholkman method is a continuous generalization of the Duncan method (which only provided estimates for four wavelengths). The Scholkman method relies on th following formula:

    .. math::
        DPF(\\lambda, Age) = \\alpha + \\beta Age^\\gamma + \\delta \\lambda^3 + \\epsilon \\lambda^2 + \\zeta  \\lambda


    References
    ----------
    Scholkmann, F., & Wolf, M. (2013). General equation for the differential pathlength factor of the frontal human head depending on wavelength and age. Journal of biomedical optics, 18(10), 105004-105004.

    Duncan, A., Meek, J. H., Clemence, M., Elwell, C. E., Tyszczuk, L., Cope, M., & Delpy, D. (1995). Optical pathlength measurements on adult head, calf and forearm and the head of the newborn infant using phase resolved optical spectroscopy. Physics in Medicine & Biology, 40(2), 295.

    Examples
    --------

    >>> dpf(0.583, (760, 850), "scholkman")
    """

    if not method in ["scholkman", "duncan"]:
        raise ValueError("`method` must be one of 'sholkman' or 'duncan'")

    if not len(wavelengths) == 2:
        raise ValueError(
            "`wavelengths` must be one an integer vector of length 2")

    if method == "scholkman":
        return tuple([dpf_scholkman(age, wavelength=w) for w in wavelengths])

    # Duncan Method
    if method == "duncan":
        return tuple([dpf_duncan(age, wavelength=w) for w in wavelengths])


def dpf2ppf(dpf: float, pvc: float = 1) -> float:
    """
    Convert Differential Pathlength Factor (DPF) to PPF.

    In line with other NIRS-related software, MNE-NIRS uses PPF instead of DPF to apply the modified Beer-Lambert Law (mBLL). MNE-NIRS defaults this value to 6, in line with Homer2, but this value can be calculated from the DPF.

    Parameters
    ----------
    dpf : float
        Differential Pathlength Factor (DPF).
    pvc: float
        Partial Volume Correction (defaults to 1).

    Returns
    -------
    float
        Calculated PPF.
    """
    ppf = dpf / pvc
    return ppf


def preprocess_participant(file: str,
                           ppf: float = 0.1,
                           verbose: bool = False,
                           plot: bool = False):
    """
    Preprocess a single participants' NIRS recording.

    This function takes the path of a SNIRF file and performs the following steps:

    1) Reads the raw intensity values, cleans the annotations
    2) Evaluates the quality of the signal (Peak Power) and flags bad channels
    3) Converts to optical densities
    4) Performs motion arctifact correction (temporal derivative distribution repair)
    5) Applies a band-pass filter to correct physiological arctifacts
    6) Converts to HbO and HbR concentration changes using the modified Beer-Lambert law
    6) Segments the signal into epochs and flags bad epochs
    Parameters
    ----------

    file : str
        Path to the SNIRF file of the participant.
    ppf : float
        PPF to apply the modified Beer Lambert Law (mBLL) (defaults to 0.1).
    verbose : bool
        Should the function print logs? (defaults to `False`).
    plot : bool
        Should the function plot intermediate outcomes? (defaults to `False`).

    Returns
    -------
    mne.Raw
        Heamodynamic response
    mne.Epochs
        Epochs of the haemodynamic response

    """
    # import raw intensity
    raw_intensity = mne.io.read_raw_snirf(file, preload=False, verbose=verbose)

    # clean annotations
    raw_intensity.annotations.rename({'1.0': 'Silence',
                                      '2.0': 'Noise',
                                      '3.0': 'Speech'})
    raw_intensity.annotations.delete(
        raw_intensity.annotations.description == '15.0')
    raw_intensity.annotations.set_durations(5)

    events, event_dict = mne.events_from_annotations(
        raw_intensity, verbose=verbose)
    raw_intensity.reorder_channels(np.sort(raw_intensity.ch_names))

    # optical density
    raw_od = mne.preprocessing.nirs.optical_density(
        raw_intensity, verbose=verbose)

    # scalp coupling index
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, 0.7, 1.45)
    raw_od.info['bads'] = [raw_od.ch_names[i]
                           for i in range(len(raw_od.ch_names)) if sci[i] < 0.8]

    # correct motion arctifacts
    t_od = temporal_derivative_distribution_repair(raw_od)

    # short channel regression
    enhanced_od = signal_enhancement.short_channel_regression(t_od)

    # hemodynamic response (modified Beer-Lamber law)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(enhanced_od, ppf=ppf)
    raw_haemo = channels.get_long_channels(raw_haemo)

    # negative correlation enhancenent
    anti_haemo = signal_enhancement.enhance_negative_correlation(raw_haemo)

    # band-pass filter
    f_haemo = anti_haemo.filter(l_freq=0.01,
                                h_freq=0.7,
                                l_trans_bandwidth=0.005,
                                h_trans_bandwidth=0.3,
                                verbose=False)

    # epoching
    reject_criteria = dict(hbo=100e-6)
    tmin, tmax = -3, 14
    epochs = mne.Epochs(f_haemo,
                        events,
                        event_id=event_dict,
                        tmin=tmin, tmax=tmax,
                        reject=reject_criteria,
                        proj=True,
                        baseline=(None, 0),
                        preload=True,
                        detrend=1,
                        verbose=False)
    if verbose:
        epochs.drop_log

    return f_haemo, epochs


if __name__ == "__main__":
    from rich.progress import track
    from glob import glob
    import os
    import pandas as pd
    # convenient function from the MNE-NIRS package to locate the data folder
    datapath = datasets.block_speech_noise.data_path()
    # list all snirf files in the folder
    snirf_files = glob(os.path.join(datapath, "**", "*.snirf"), recursive=True)
    epochs_group = pd.DataFrame()
    for f in track(snirf_files, description="Processing"):
        # analyse data and return both ROI and channel results
        raw_haemo, epochs = preprocess_participant(f)
        e = epochs.to_data_frame()
        e["file"] = os.path.basename(f)
        epochs_group = pd.concat([epochs_group, e])
    epochs_group.to_csv(os.path.join("out", "hdr.csv"), index=False)

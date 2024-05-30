import os
import glob
import numpy as np
import mne

from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
from mne_nirs import datasets, visualisation


# get brain files
datapath = datasets.block_speech_noise.data_path()
fs_path = "\\\\wsl$\\Ubuntu\\usr\\local\\freesurfer\\7.4.1"
subjects_dir = datapath + "\\subjects"
dataset = BIDSPath(root=subjects_dir,
                   suffix="nirs",
                   extension=".snirf",
                   subject="04",
                   task="AudioSpeechNoise",
                   datatype="nirs",
                   session="01")

# Download anatomical locations
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
labels_combined = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)
mne.datasets.fetch_fsaverage(subjects_dir)

# import raw intensity and clean annotations
snirf_files = glob.glob(os.path.join(
    datapath, "**", "*.snirf"), recursive=True)
raw_intensity = mne.io.read_raw_snirf(snirf_files[0])
raw_intensity.annotations.rename({'1.0': 'Silence',
                                  '2.0': 'Noise',
                                  '3.0': 'Speech'})
raw_intensity.annotations.delete(
    raw_intensity.annotations.description == '15.0')
raw_intensity.annotations.set_durations(5)

# plot brain
mne.viz.set_3d_backend("pyvistaqt", verbose=None)

fs_path = "\\\\wsl$\\Ubuntu\\usr\\local\\freesurfer\\7.4.1"
subjects_dir = fs_path + "\\subjects"
mne.datasets.fetch_fsaverage(subjects_dir)
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                        accept=True)
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
labels_combined = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)

brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, background="w", cortex="0.5"
)

brain.add_sensors(raw_intensity.info,
                  trans="fsaverage",
                  fnirs=["channels", "pairs", "sources", "detectors"])


def get_ch_index(channels: list[str], ch_names: list[str]) -> list[int]:
    s = []
    for x in channels:
        for w in [" 760", " 850"]:
            s.append(x + w)
    return [ch_names.index(x) for x in s]


view_map = {
    'left-lat': 1+np.array(get_ch_index(left_stg, raw_intensity.ch_names)),
    'caudal': 1+np.array(get_ch_index(occipital, raw_intensity.ch_names)),
    'right-lat': 1+np.array(get_ch_index(right_stg, raw_intensity.ch_names))
}

view_map = {
    'left-lat': np.arange(1, 23),
    'caudal': np.arange(24, 46),
    'right-lat': np.arange(46, 68)
}

fig_montage = visualisation.plot_3d_montage(
    raw_intensity.info, view_map=view_map, subjects_dir=subjects_dir)

brain.show_view(azimuth=20, elevation=60, distance=400)

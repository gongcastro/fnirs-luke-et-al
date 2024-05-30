from rich.progress import track
from rich.console import Console, Text
import mne
import mne_nirs
import snirf
import matplotlib.pylab as plt
import numpy as np
from mne_nirs.io import write_raw_snirf
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix
np.random.seed(1)


def simulate_dataset(n_channels: int = 10):
    channels = np.arange(n_channels)
    # stimulus duration: syllable duration, 3 syllables per word, in seconds
    # events (no pauses between blocks are implemented)
    condition_labels = ["A", "Silence", "B", "Silence"]
    words_per_block = 10
    n_blocks = 28
    block_duration = np.sum(
        [stim_duration] * words_per_block +
        isi * int(words_per_block / 2))
    ibi = [25, 35]  # inter-block interval (silence)
    ibi_series = ibi * n_blocks
    run_duration = np.sum([block_duration] * n_blocks + ibi * n_blocks)

    n_blocks_per_condition = 14
    stim_series = np.concatenate([np.repeat(condition_labels, words_per_block)
                                  for _ in np.arange(int(n_blocks/2))])
    np.random.shuffle(block_series)

    events = condition_labels * 14
    channel_names = ["Ch" + str(x) for x in np.arange(n_channels)] * 2
    channel_names = [
        x + y for x in channel_names for y in [" 760", " 850"]]

    blocks = []

    for idx, ch in enumerate(channel_names):
        console = Console()
        text = Text(f"Simulating channel {ch} ({idx+1}/{len(channel_names)})")
        text.stylize("bold magenta")
        console.print(text)
        blocks.append(simulate_channel(ch, events))

    blocks = blocks[0].add_channels(blocks[1:len(blocks)])
    return blocks


def simulate_channel(ch_name: str,
                     events: list[str],
                     stim_duration: float = 270 * 3 / 1000,
                     run_duration: float = 1405.8):
    channel = []
    for e in track(events, description=f"Generating blocks"):
        if "760" in ch_name:
            amp = 0
        else:
            if e == "Silence":
                amp = 0
            elif e == "A":
                amp = 0.15
            else:
                amp = 0.10

        stm_dur = np.random.choice(
            [25, 35]) if c == "Silence" else stim_duration

        block = mne_nirs.simulation.simulate_nirs_raw(
            sfreq=10,
            sig_dur=run_duration,
            stim_dur=stm_dur,
            amplitude=amp,
            isi_min=1.5,
            isi_max=0.5,
            ch_name=ch_name,
            annot_desc=e)
        block = block.crop(tmin, block_duration)

        # add noise
        cov = mne.Covariance(np.ones(1) * 1e-11,
                             block.ch_names,
                             block.info['bads'],
                             block.info['projs'],
                             nfree=0)
        filterbank = [
            1., -0.58853134, -0.29575669,  -0.52246482, 0.38735476, 0.02428681]
        block = mne.simulation.add_noise(block, cov,
                                         iir_filter=filterbank,
                                         verbose=0)
        channel.append(block)

    channel = mne.concatenate_raws(channel)
    return channel


if __name__ == "__main__":
    import os
    blocks = simulate_dataset(n_channels=5)
    # write_raw_snirf(blocks.load_data(), os.path.join(
    #    "data", "simulation.snirf"))
    blocks.plot(show_scrollbars=False)

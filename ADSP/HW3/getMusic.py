import numpy as np
from scipy.io.wavfile import write
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--score", type=str,
    help="Please input your score list 1 Do 2 Re 3 Mi 4 Fa 5 so 6 La 7 Si -999 is rest(comma separate)")
parser.add_argument(
    "--beat", type=str, default="default_beat",
    help="Please assign list beat of score(comma separate/same size with beat)")
parser.add_argument(
    "--sharp", type=str, default="default_sharp",
    help="Please assign list sharp use 0 1 to represent whether sharp corresponding score(comma separate/same size with beat)")
parser.add_argument(
    "--num_octave", type=int, default="0",
    help="Get high octave frequency music")
parser.add_argument(
    "--name", type=str, default="music",
    help="Please assign your output wav name")
parser.add_argument(
    "--bpm", type=int, default="120",
    help="Please assign your music BPM")
args = parser.parse_args()
# input_tone = np.array([5,3,3,4,2,2,1,2,3,4,5,-999,5,-999,5]).astype("int")
input_tone = np.array(args.score.split(",")).astype("int")
input_tone = input_tone + 8 * args.num_octave
if args.beat == "default_beat":
    beat = np.ones(len(input_tone))
else:
    beat = np.array(args.beat.split(",")).astype("float")
if args.sharp == "default_sharp":
    sharp = np.zeros(len(input_tone))
else:
    sharp = np.array(args.sharp.split(",")).astype("int")
# input_tone = np.array([7,4,4,5,2,2,0,2,4,5,7,7,7])
wavname = args.name
dict_tone_mapping = {
    -999: -1000,
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 7,
    6: 9,
    7: 11,
}
# modify_tone = np.array(list(map(dict_tone_mapping.get, input_tone)))
bpm = args.bpm
sample_rate = 44100
base_frequency = 131.32
per_freq_ratio = 2**(1/12)
result = np.array([]).astype("float32")
for i in range(len(input_tone)):
    if input_tone[i] == -999:
        samples = (np.sin(2 * np.pi * np.arange(sample_rate * 60/bpm*beat[i])
            * 0*(per_freq_ratio**modified_tone) / sample_rate)).astype(np.float32)
    else:    
        modified_tone = dict_tone_mapping[int(input_tone[i] % 8)] + input_tone[i]//8 * 12 + sharp[i]
        samples = (np.sin(2 * np.pi * np.arange(sample_rate * 60/bpm*beat[i])
            * base_frequency*(per_freq_ratio**modified_tone) / sample_rate)).astype(np.float32)
    result = np.r_[result, samples]
write('%s.wav' % wavname, sample_rate, result)
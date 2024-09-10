# Generates drums based on audio file
# Constraints:
# music should start at index 0, no lead time or silence before starting
# default tempo is 120BPM, 4/4
# Each chunk of drum is generated for 4 bars (default) so don't expect major changes in this timeframe

import argparse
from pydub import AudioSegment
import numpy as np
from input_audio_analysis import get_tags_from_audio
from musical_ai import get_prompt_from_tags
from drum_generator import generate_audio_from_prompt

parser = argparse.ArgumentParser(description='Generate a drum part for a given audio file. Default tempo 80BPM, changes every 4 bars')
parser.add_argument('--input', default='input.mp3', type=str, help='Input audio file as an MP3, default is input.mp3')
parser.add_argument('--output', default='output.wav', type=str, help='Output audio file as an wav, default is output.wav')
parser.add_argument('--tempo', default=120, type=int, choices=range(40, 200), help='Tempo of your track')
parser.add_argument('--bars', default=4, type=int, choices=range(1, 16), help='Bars per generation chunk')

def generate_drum_track_from_audio_file():

    args = parser.parse_args()

    print("*** Drums generator ***")
    print("Will generate a drum part from the following configuration:")
    print(f"Input file:{args.input}")
    print(f"Tempo: {args.tempo}")
    print(f"Bars per generation chunk : {args.bars}")
    print(f"Output file: {args.output}")

    input = args.input
    tempo = args.tempo
    bars = args.bars
    output = args.output

    # load file
    audio_track = AudioSegment.from_mp3(input)
    audio_track.set_channels(1)

    # Export the audio data to raw audio
    raw_data = audio_track.raw_data

    # Convert the raw audio data to a NumPy array
    samples = np.frombuffer(raw_data, dtype=np.int32)

    # Normalize the array to be in the range of -1 to 1
    samples = samples.astype(np.float32) / np.iinfo(np.int32).max

    # process each x bars
    print(len(samples))
    print(audio_track.frame_rate)
    print(audio_track.channels)    
    bars_in_samples = int((( 1 / tempo * 60) * 4 * bars ) * audio_track.frame_rate)
    print(bars_in_samples)

    audio_output = AudioSegment.empty()

    block_processed = 0
    max_blocks_to_process = 6

    for pos in range(0, len(samples), bars_in_samples):  

        print(f"processing chunk at {pos} samples")

        # get tags for the bars
        tags = get_tags_from_audio(samples[pos:pos+bars_in_samples])
        print(tags)
        if len(tags) == 0:
            continue

        # generate prompt from the tags
        prompt = get_prompt_from_tags(tags, tempo, bars)
        print(f"Claude suggested to generate audio with this prompt: {prompt}")

        # generate drum audio for the x bars from the prompt
        audio_chunk = generate_audio_from_prompt(prompt, tempo, bars)

        # add it to the output
        audio_output += audio_chunk

        block_processed += 1

        if block_processed >= max_blocks_to_process:
            print(f"Skipping at block #{block_processed}")
            break

    # save the output
    AudioSegment.export(audio_output, format="wav", out_f=output)

    print("Done!")

if __name__ == "__main__":
    generate_drum_track_from_audio_file()

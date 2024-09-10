import onnxruntime as ort
import torch
from pydub import AudioSegment
import pydub
import numpy as np

# This is a list of labels that the model can classify, you can adjust it to your needs
MTT_LABELS = [
    #'guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral'
            "rock", 
        "pop",
        "jazz", 
        "metal", 
        "classical", 
        "country",
        "funk",
        "slow",
        "fast",
        "chords",
        "riff",
        "solo",
        "ambiant",
        "quiet",
        "calm",
        "intense",
        "heavy",
        "repetitive",
        "diatonique",
        "atonal",
        "clean",
        "crunch",
        "reverb",
        "overdrive",
        "distortion",
        "fuzz",
        "echoes",
        "blury",
        "cathedral",
        "room"
    ];

input_sample_length = 59049

options = ort.SessionOptions()
options.enable_profiling=True
print("Loading classification model")
ort_sess = ort.InferenceSession('./models/clmr_sample-cnn.onnx', sess_options=options)
print("Loaded classification model")

def softmax(arr):
    C = np.max(arr)
    exp_values = np.exp(arr - C)
    sum_exp_values = np.sum(exp_values)
    return exp_values / sum_exp_values

def sort_chart(all_data):

    # Sort the list by the data value in descending order
    all_data.sort(key=lambda x: x['data'], reverse=True)

    # Split them back into sorted labels and sorted data
    sorted_labels = [item['label'] for item in all_data][:10]
    sorted_data = [item['data'] for item in all_data][:10]

    # Normalize the data values to be between 0 and 1
    min_value = min(sorted_data)
    max_value = max(sorted_data)
    normalized_data = [(x - min_value) / (max_value - min_value) for x in sorted_data]

    return {
        'sorted_data': sorted_data,
        'sorted_labels': sorted_labels
    }




def get_tags_from_audio(samples: np.ndarray) -> list:

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name

    analysis_array = []

    # divide the samples in chunks aligned with model inputs and get the tags from adding up all the tags in each chunk
    for i in range(0,len(samples),input_sample_length):

        start = i
        samples_tmp = samples[start:start+input_sample_length]

        # not using the remaining block
        if len(samples_tmp) < input_sample_length:
            break

        # prepare inputs. a tensor needs its corresponding numpy array as data
        audio_tensor = samples_tmp.reshape(1, 1, input_sample_length).astype(np.float32)

        # prepare feeds. use model input names as keys.
        feeds = { 'audio': audio_tensor }

        # run inference
        results = ort_sess.run(["representation"], feeds)

        new_results = results[0][0]
        dataset = softmax(new_results)
        analysis_array.append(dataset)

        # Combine dataset and labels into a list of dictionaries
        all_data = [{'label': MTT_LABELS[l], 'data': dataset[l]} for l in range(len(MTT_LABELS))]

        final_results = sort_chart(all_data)

    if len(analysis_array) == 0:
        return []
    
    np_analysis_array = np.array(analysis_array)

    means = np.sum(np_analysis_array, axis=0)
    dataset = softmax(means)
    all_data = [{'label': MTT_LABELS[l], 'data': dataset[l]} for l in range(len(MTT_LABELS))]
    final_results = sort_chart(all_data)

    return final_results['sorted_labels'][0:3]


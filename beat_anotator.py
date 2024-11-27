import json
import os
import mido

def beat_duration(signature, tempo):
    """
    Calculates the duration of a beat based on the time signature and BPM.
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duração de uma semínima
    return beat_length * (4 / denominator)

def calculate_expected_beats(measures, time_signature):
    """
    Calculates the total number of expected beats based on the measures.
    """
    numerator, _ = map(int, time_signature.split('/'))
    return measures * numerator

def extract_midi_beats(midi_file, tempo):
    """
    Extracts the beat times from the MIDI file.
    """
    midi = mido.MidiFile(midi_file)
    beats = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)

    for msg in midi:
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == 'note_on' and msg.velocity > 0:
            beats.append(round(time_elapsed, 3))
    return beats

def generate_annotations(instance_dir):
    """
    Generates the beat annotations using the MIDI and JSON files of the instance.
    """
    json_file = os.path.join(instance_dir, f"{os.path.basename(instance_dir)}.json")
    output_file = os.path.join(instance_dir, f"{os.path.basename(instance_dir)}-beats.txt")

    with open(json_file, 'r') as f:
        data = json.load(f)

    arrangement = data["arrangement"]
    measures = data["measures"]
    time_signatures = data["time_signatures"]
    tempo = data["tempo"]
    transitions = data["transitions"]

    annotations = []
    current_time = 0.0

    for i, part in enumerate(arrangement):
        signature = time_signatures.get(part)
        part_measures = measures.get(part, 0)
        midi_filename = f"{os.path.basename(instance_dir)}-{part}-beat.mid"
        midi_path = os.path.join(instance_dir, midi_filename)

        theoretical_beats = []
        part_beat_duration = beat_duration(signature, tempo)

        # Generate theoretical beats
        for measure in range(part_measures):
            for beat in range(int(signature.split('/')[0])):  # Numerador = beats por compasso
                theoretical_beats.append(round(current_time, 3))
                annotations.append(f"{current_time:.3f}\t{beat + 1}")
                current_time += part_beat_duration

        # Validating with MIDI beats
        if os.path.exists(midi_path):
            midi_beats = extract_midi_beats(midi_path, tempo)
            expected_beats = calculate_expected_beats(part_measures, signature)

            if len(midi_beats) < expected_beats:
                print(f"Warning: Part '{part}' - Missing beats in MIDI.")
            elif len(midi_beats) > expected_beats:
                print(f"Warning: Part '{part}' - Excess beats in MIDI.")

        # Adjusts the time according to the transition
        if i < len(transitions) - 1:
            next_transition_time = transitions[i + 1][1]
            current_time = max(current_time, next_transition_time)

    # Saves the annotations to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(annotations))

    print(f"Annotations saved to: {output_file}")

# Example of usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python beat_anotator.py <instance dir>")
    else:
        generate_annotations(sys.argv[1])

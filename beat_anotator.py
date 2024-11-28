import json
import os
import mido
from typing import List, Tuple

def beat_duration(signature: str, tempo: int) -> float:
    """
    Calculates the duration of a beat based on the time signature and BPM.
    """
    numerator, denominator = map(int, signature.split('/'))
    beat_length = 60 / tempo  # Duration of a quarter note
    return beat_length * (4 / denominator)

def calculate_expected_beats(measures: int, time_signature: str) -> int:
    """
    Calculates the total number of expected beats based on the measures.
    """
    numerator, _ = map(int, time_signature.split('/'))
    return measures * numerator

def extract_midi_beats(midi_file: str, tempo: int, start_time: float) -> List[float]:
    """
    Extracts the beat times from the MIDI file, adjusting for the start time.
    """
    midi = mido.MidiFile(midi_file)
    beats = []
    time_elapsed = 0.0
    ticks_per_beat = midi.ticks_per_beat
    tempo_us = mido.bpm2tempo(tempo)

    for msg in midi:
        time_elapsed += mido.tick2second(msg.time, ticks_per_beat, tempo_us)
        if msg.type == 'note_on' and msg.velocity > 0:
            beats.append(round(time_elapsed + start_time, 3))
    return sorted(beats)

def compare_beats(theoretical_beats: List[float], midi_beats: List[float]) -> Tuple[bool, str]:
    """
    Compares the theoretical beats with the beats extracted from the MIDI and verifies the alignment.
    """
    if not midi_beats:
        return False, "No beats found in MIDI file."

    if len(theoretical_beats) != len(midi_beats):
        return False, f"Mismatch in number of beats. Expected {len(theoretical_beats)}, found {len(midi_beats)}."

    tolerance = 0.01  # 10 milliseconds
    for i, (theoretical, midi) in enumerate(zip(theoretical_beats, midi_beats)):
        if abs(theoretical - midi) > tolerance:
            return False, f"Misalignment detected at beat {i + 1}. Theoretical: {theoretical:.3f}, MIDI: {midi:.3f} (diff: {abs(theoretical - midi):.3f})."

    return True, "Beats are aligned."

def generate_annotations(instance_dir: str) -> None:
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
        signature = time_signatures[part]
        part_measures = measures[part]
        midi_filename = f"{os.path.basename(instance_dir)}-{part}-beat.mid"
        midi_path = os.path.join(instance_dir, midi_filename)

        theoretical_beats = []
        part_beat_duration = beat_duration(signature, tempo)

        # Generate theoretical beats
        for measure in range(part_measures):
            for beat in range(int(signature.split('/')[0])):  # Numerator = beats per measure
                theoretical_beats.append(round(current_time, 3))
                annotations.append(f"{current_time:.3f}\t{beat + 1}")
                current_time += part_beat_duration

        # Validate beats with MIDI
        if os.path.exists(midi_path):
            midi_beats = extract_midi_beats(midi_path, tempo, transitions[i][1])
            is_valid, message = compare_beats(theoretical_beats, midi_beats)
            if not is_valid:
                print(f"Warning: Part '{part}' - {message}")

        # Adjust time according to the transition
        if i < len(transitions) - 1:
            current_time = float(transitions[i + 1][1])

    # Save the annotations to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(annotations))

    print(f"Annotations saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python beat_annotator.py <instance_dir>")
        sys.exit(1)
    
    generate_annotations(sys.argv[1])

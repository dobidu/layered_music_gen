import json
import os
import mido

def generate_annotations(instance_dir):
    """
    Generates beat annotations using the MIDI and JSON files of the instance.
    """

    # Checks if the instance directory exists
    if not os.path.exists(instance_dir):
        print(f"Diretório da instância não encontrado: {instance_dir}")
        return
    
    # Removes the directory name slash (if any)
    if instance_dir.endswith(os.sep):
        instance_dir = instance_dir[:-1]
    
    json_file = os.path.join(instance_dir, f"{os.path.basename(instance_dir)}.json")
    midi_files = [f for f in os.listdir(instance_dir) if f.endswith("-beat.mid")]

    # Defines the output file name
    output_file = os.path.join(instance_dir, f"{os.path.basename(instance_dir)}-beats.txt")

    # Loads the JSON file of the instance
    with open(json_file, 'r') as f:
        data = json.load(f)

    arrangement = data["arrangement"]
    measures = data["measures"]
    time_signatures = data["time_signatures"]
    tempo = data["tempo"]
    transitions = data["transitions"]

    annotations = []  # List to store the annotations

    def beat_duration(signature, tempo):        
        """
        Calculates the duration of a beat based on the time signature and BPM.
        """
        numerator, denominator = map(int, signature.split('/'))
        beat_length = 60 / tempo  # Duração de uma semínima
        return beat_length * (4 / denominator)

    # Processes each part of the arrangement
    current_time = 0.0
    for i, part in enumerate(arrangement):
        signature = time_signatures.get(part)
        part_measures = measures.get(part, 0)
        midi_filename = f"{os.path.basename(instance_dir)}-{part}-beat.mid"
        midi_path = os.path.join(instance_dir, midi_filename)

        if not os.path.exists(midi_path):
            print(f"MIDI file not found: {midi_filename}")
            continue

        part_beat_duration = beat_duration(signature, tempo)

        # Anotates each beat for this part
        for measure in range(part_measures):
            for beat in range(int(signature.split('/')[0])):  # Numerador = beats por compasso
                annotations.append(f"{current_time:.3f}\t{beat + 1}")
                current_time += part_beat_duration

        # Adjusts the time according to the transition
        if i < len(transitions) - 1:
            next_transition_time = transitions[i + 1][1]
            current_time = max(current_time, next_transition_time)

    # Saves the annotations to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(annotations))

    print(f"Beat anotation file saved at: {output_file}")

# Exemple usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python beat_anotator.py <instance dir>")
    else:
        generate_annotations(sys.argv[1])

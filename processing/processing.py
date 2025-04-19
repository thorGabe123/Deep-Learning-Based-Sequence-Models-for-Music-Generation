import os
import numpy as np
import pretty_midi
from note import MIDI_note
from pathlib import Path
import shutil
import re
import configs.common as cc

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def preprocess_midi_files(midi_folder, preprocess_folder):
    midi_paths = list(find_files_by_extensions(midi_folder, ['.mid', '.midi']))
    os.makedirs(midi_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    for path in midi_paths:
        path_parts = Path(path).parts
        band_name = path_parts[-2]
        song_name = Path(path).stem

        os.makedirs(os.path.join(preprocess_folder, band_name), exist_ok=True)

        new_path = os.path.join(preprocess_folder, band_name, song_name)
        
        print(' ', end='[{}]'.format(path), flush=True)

        # Check if the file already exists or if a file with a similar name exists
        if os.path.exists(new_path + '.npy'):
            continue

        # Check if any file in the preprocess_folder contains the new_path as a substring
        if re.search(r'\.\d+$', new_path):
            continue

        try:
            midi_notes = extract_midi(path)
            if len(midi_notes) == 0:
                continue
            token_seq = encode(midi_notes)
            np.save(new_path + '.npy', token_seq)
        except:
            continue

def extract_midi(path):
    mid = pretty_midi.PrettyMIDI(midi_file=path)
    tempo_times, tempo_bpm = mid.get_tempo_changes()
    end_time = mid.get_end_time()
    tempo_times = np.append(tempo_times, end_time)

    midi_notes = []
    for inst in mid.instruments:
        if not inst.is_drum:
            channel = int(inst.program)
        else:
            channel += 128
        for n in inst.notes:
            idx = next((i for i, t in enumerate(tempo_bpm) if tempo_times[i] <= n.start < tempo_times[i + 1]))
        
            midi_notes.append(MIDI_note(pitch=abs(n.pitch), 
                                        time_start=abs(n.start), 
                                        time_end=abs(n.end), 
                                        dynamic=abs(n.velocity), 
                                        channel=abs(channel), 
                                        tempo=round(tempo_bpm[idx])))

    midi_notes = list(set(midi_notes))
    midi_notes = sorted(midi_notes, key=lambda note: note.time_start)

    return midi_notes

def note_to_midi(midi_notes, output_path):
    midi_object = pretty_midi.PrettyMIDI()

    # Group notes by channel if you have multiple channels
    channel_to_notes = {}
    for note in midi_notes:
        channel_to_notes.setdefault(note.channel, []).append(note)

    for channel, notes in channel_to_notes.items():
        if channel >= 128:
            instrument = pretty_midi.Instrument(program=channel-128, is_drum=True)
        else:
            instrument = pretty_midi.Instrument(program=channel, is_drum=False)
        for note in notes:
            pm_note = pretty_midi.Note(
                velocity=int(note.dynamic),
                pitch=int(note.pitch),
                start=float(note.time_start),
                end=float(note.time_end)
            )
            instrument.notes.append(pm_note)
        midi_object.instruments.append(instrument)

    update_tempo(midi_object, midi_notes)

    os.makedirs("midi", exist_ok=True)

    midi_object.write(f"midi/{output_path}")

def adjust_note_time(midi_notes):
    res_per_beat = cc.config.resolution.bar_res
    current_beats = 0
    prev_time = 0
    prev_tempo = midi_notes[0].tempo
    for idx, n in enumerate(midi_notes):
        resolution = 60 / prev_tempo / res_per_beat
        current_beats += (n.time_start - prev_time) / resolution
        future_beats = current_beats + (n.time_end - n.time_start) / resolution
        prev_time = n.time_start
        prev_tempo = n.tempo
        midi_notes[idx].time_start = int(current_beats)
        if int(future_beats) == int(current_beats):
            midi_notes[idx].time_end = int(current_beats) + 1
        else:
            midi_notes[idx].time_end = int(future_beats)

def encode(midi_notes):
    adjust_note_time(midi_notes)

    token_seq = []
    time_prev = 0
    time_delta_prev = 0
    for idx, m in enumerate(midi_notes):
        dynamic = cc.start_idx['dyn'] + min(m.dynamic, cc.config.discretization.dyn - 1)
        pitch = cc.start_idx['pitch'] + min(m.pitch, cc.config.discretization.pitch - 1)
        length = cc.start_idx['length'] + min(m.time_end - m.time_start, cc.config.discretization.length - 1)
        time_delta = cc.start_idx['time'] + min(m.time_start - time_prev, cc.config.discretization.time - 1)
        channel = cc.start_idx['channel'] + min(m.channel, cc.config.discretization.channel - 1)
        tempo = cc.start_idx['tempo'] + min(m.tempo, cc.config.discretization.tempo - 1)

        token_seq.append(channel)
        token_seq.append(pitch)
        token_seq.append(dynamic)
        token_seq.append(length)
        if time_delta_prev != time_delta:
            token_seq.append(time_delta)
        token_seq.append(tempo)
        time_prev = m.time_start
        time_delta_prev = time_delta

    return token_seq

def revert_note_time(midi_notes):
    res_per_beat = cc.config.resolution.bar_res
    prev_time = 0
    prev_beat = 0
    prev_tempo = midi_notes[0].tempo
    for idx, n in enumerate(midi_notes):
        # Calculate time_start and time_end
        resolution = 60 / prev_tempo / res_per_beat
        time_start = prev_time + (n.time_start - prev_beat) * resolution
        time_end = time_start + (n.time_end - n.time_start) * resolution

        # Update current time for the next note
        prev_time = time_start
        prev_beat = n.time_start
        prev_tempo = n.tempo
        
        midi_notes[idx].time_start = time_start
        midi_notes[idx].time_end = time_end

def decode(token_seq):
    decoded_notes = []
    prev_time = 0

    dynamic = None
    pitch = None
    length = None
    time_delta = 0
    channel = None
    tempo = None

    for token in token_seq:
        if token < cc.start_idx['dyn']:
            pitch = token - cc.start_idx['pitch']
        elif cc.start_idx['dyn'] <= token and token < cc.start_idx['length']:
            dynamic = token - cc.start_idx['dyn']
        elif cc.start_idx['length'] <= token and token < cc.start_idx['time']:
            length = token - cc.start_idx['length']
        elif cc.start_idx['time'] <= token and token < cc.start_idx['channel']:
            time_delta = token - cc.start_idx['time']
        elif cc.start_idx['channel'] <= token and token < cc.start_idx['tempo']:
            channel = token - cc.start_idx['channel']
        elif cc.start_idx['tempo'] <= token:
            tempo = token - cc.start_idx['tempo']

        if all([x is not None for x in [dynamic, pitch, length, time_delta, channel, tempo]]):
            note = MIDI_note(dynamic=int(dynamic),
                pitch = int(pitch),
                time_start = float(prev_time + time_delta),
                time_end = float(prev_time + time_delta + length),
                channel = int(channel),
                tempo = float(tempo)
            )
            decoded_notes.append(note)
            dynamic = None
            pitch = None
            length = None
            channel = None
            tempo = None
            
            prev_time = prev_time + time_delta

    revert_note_time(decoded_notes)
    return decoded_notes

def update_tempo(mid, decoded_notes):
    new_tick_scales = []
    prev_tempo = 0
    for note in decoded_notes:
        if prev_tempo != note.tempo:
            tempo = 60.0/(note.tempo*mid.resolution)
            time = mid.time_to_tick(note.time_start)
            new_tick_scales.append((time, tempo))
            prev_tempo = note.tempo
    mid._tick_scales = new_tick_scales

def get_directory_size(directory):
    """Calculate the total size of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def get_filenames_sorted_by_size(folder_path):
    # List all directories in the folder
    directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # Calculate the size of each directory
    directories_with_size = [(d, get_directory_size(os.path.join(folder_path, d))) for d in directories]
    
    # Sort directories by size
    directories_with_size.sort(key=lambda x: x[1], reverse=True)
    
    # Extract the sorted directory names
    sorted_directories = [d[0] for d in directories_with_size]
    
    return sorted_directories

def remove_irrelevant_directories(folder_path, relevant_files):
    # List all directories in the folder
    directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # Remove directories not in the relevant_files list
    for directory in directories:
        if directory not in relevant_files:
            dir_path = os.path.join(folder_path, directory)
            print(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
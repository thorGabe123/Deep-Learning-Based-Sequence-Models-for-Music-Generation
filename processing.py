import os
import numpy as np
import pretty_midi
from note import MIDI_note
from pathlib import Path
from config import *

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
    return files

def preprocess_midi_files(midi_folder, preprocess_folder):
    midi_paths = list(find_files_by_extensions(midi_folder, ['.mid', '.midi']))
    os.makedirs(midi_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    for path in midi_paths:
        file_name = Path(path).stem
        new_path = os.path.join(preprocess_folder, file_name)
        
        print(' ', end='[{}]'.format(path), flush=True)
        if os.path.exists(new_path + '.npy'):
            continue

        midi_notes = extract_midi(path)
        if len(midi_notes) == 0:
            continue
        token_seq = encode(midi_notes)
        np.save(new_path, token_seq)

def extract_midi(path):
    mid = pretty_midi.PrettyMIDI(midi_file=path)
    tempo_times, tempo_bpm = mid.get_tempo_changes()
    end_time = mid.get_end_time()
    tempo_times = np.append(tempo_times, end_time)

    midi_notes = []
    for inst in mid.instruments:
        channel = inst.program
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

def adjust_note_time(midi_notes):
    res_per_beat = BAR_RES
    current_beats = 0
    prev_time = 0
    prev_tempo = midi_notes[0].tempo
    for idx, n in enumerate(midi_notes):
        resolution = 60 / prev_tempo / res_per_beat
        current_beats += round((n.time_start - prev_time) / resolution)
        future_beats = current_beats + round((n.time_end - n.time_start) / resolution)
        prev_time = n.time_start
        prev_tempo = n.tempo
        midi_notes[idx].time_start = current_beats
        midi_notes[idx].time_end = future_beats

def encode(midi_notes):
    adjust_note_time(midi_notes)

    token_seq = []
    time_prev = 0
    for idx, m in enumerate(midi_notes):
        dynamic = START_IDX['DYN_RES'] + min(m.dynamic, DYN_RES - 1)
        pitch = START_IDX['PITCH_RES'] + min(m.pitch, PITCH_RES - 1)
        length = START_IDX['LENGTH_RES'] + min(m.time_end - m.time_start, LENGTH_RES - 1)
        time_delta = START_IDX['TIME_RES'] + min(m.time_start - time_prev, TIME_RES - 1)
        channel = START_IDX['CHANNEL_RES'] + min(m.channel, CHANNEL_RES - 1)
        tempo = START_IDX['TEMPO_RES'] + min(m.tempo, TEMPO_RES - 1)

        token_seq.extend([dynamic, 
                          pitch, 
                          length, 
                          time_delta,
                          channel,
                          tempo])

        time_prev = m.time_start

    return token_seq

def revert_note_time(midi_notes):
    res_per_beat = BAR_RES
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

    for i in range(0, len(token_seq), 6):  # Process tokens in groups of 6
        dynamic = token_seq[i] - START_IDX['DYN_RES']
        pitch = token_seq[i + 1] - START_IDX['PITCH_RES']
        length = token_seq[i + 2] - START_IDX['LENGTH_RES']
        time_delta = token_seq[i + 3] - START_IDX['TIME_RES']
        channel = token_seq[i + 4] - START_IDX['CHANNEL_RES']
        tempo = token_seq[i + 5] - START_IDX['TEMPO_RES']

        note = MIDI_note(dynamic=dynamic,
            pitch = pitch,
            time_start = prev_time + time_delta,
            time_end = prev_time + time_delta + length,
            channel = channel,
            tempo = tempo
        )
        decoded_notes.append(note)
        prev_time = prev_time + time_delta

    revert_note_time(decoded_notes)
    return decoded_notes
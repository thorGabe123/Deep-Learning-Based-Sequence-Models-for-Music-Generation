class MIDI_note:
    def __init__(self, pitch, time_start, time_end, dynamic, channel, tempo):
        self.pitch = pitch
        self.time_start = time_start
        self.time_end = time_end
        self.dynamic = dynamic
        self.channel = channel
        self.tempo = tempo
        
    def __repr__(self):
        return (f"MIDI_note(pitch={self.pitch}, time_start={self.time_start}, "
                f"time_end={self.time_end}, dynamic={self.dynamic}, channel={self.channel}, tempo={self.tempo})")
    
    def __eq__(self, other):
        if isinstance(other, MIDI_note):
            return (self.pitch == other.pitch and
                    self.time_start == other.time_start and
                    self.time_end == other.time_end and
                    self.dynamic == other.dynamic and
                    self.channel == other.channel
                    )
        return False
    
    def __hash__(self):
        return hash((self.pitch, self.time_start, self.time_end, self.dynamic, self.channel))
    
    def note2seq(self):
        return [self.dynamic, self.pitch, self.time_end - self.time_start]
import torch
from mido import Message


class Tokeniser:
    def __init__(self, ):
        self.time_ms_max  = 60000
        self.time_max  = 6000

        self.velocity_max = 128
        self.note_max = 128
    
    @staticmethod
    def process_midi(midi):
        # In: Librosa midi file
        times = []
        msgs = []
        for i, track in enumerate(midi.tracks):
            abs_time = 0

            # remove initial track
            if i ==0:
                continue

            for msg in track:
                abs_time += msg.time
                if msg.type in ['note_on', 'note_off']:
                    msgs.append(msg)
                    times.append(round(abs_time/10) * 10)
                else:
                    continue

        times=  torch.tensor(times, dtype=torch.int32)
        return times, msgs
    
    @staticmethod
    def collect_indices_between(times, start_time, end_time):
        return [i for i, t in enumerate(times) if start_time <= t <= end_time] 

    @staticmethod
    def get_midi_between(midi, start_index=None, end_index=None, start_time=None, end_time=None):
        times, msgs = Tokeniser.process_midi(midi)

        if (start_index is not None) and (end_index is not None):
            if end_index > len(times) - 1:
                raise ValueError
            idxs = [start_index, end_index]
        elif (start_time is not None) and (end_time is not None):
            idxs = Tokeniser.collect_indices_between(times, start_time, end_time)
        else: 
            raise ValueError(f'Start_index:{start_index}')
        return times[idxs[0]:idxs[-1]], msgs[idxs[0]:idxs[-1]]
    
    def tokenise_midi(self, time_ms, velocity, note):

        assert time_ms >= 0 and time_ms < self.time_ms_max
        assert velocity >= 0 and velocity < self.velocity_max
        assert note >0 and note < self.note_max
        # To nearest 10 ms
        time = time_ms // 10
        return torch.tensor([time, self.time_max + velocity, self.time_max + self.velocity_max + note], dtype=torch.int32)
    
    def process_chunk(self, times, msgs):
        t0 = times[0]
        chunk = torch.tensor([], dtype=torch.int16)
        for t, m in zip(times, msgs):
            chunk = torch.cat([chunk, self.tokenise_midi(t-t0, m.velocity, m.note)])
        return chunk
    
    def detokenise_midi(self, token_sequence, start_time_ms = 0):
        """
        Convert a flat token sequence into a list of MIDI messages and times.
        token_sequence: 1D tensor of shape (N,) where N is a multiple of 3.
        Returns: list of (absolute_time_ms, mido.Message)
        """
        assert token_sequence.ndim == 1 and token_sequence.size(0) % 3 == 0

        messages = []
        time_ms = 0
        velocity = 64  # Default velocity
        prev_time_ms = 0
        for i in range(0, len(token_sequence), 3):
            t_token = token_sequence[i].item()
            v_token = token_sequence[i + 1].item()
            n_token = token_sequence[i + 2].item()

            # Decode absolute time (quantized to 10ms)
            time_ms = t_token * 10

            # Decode velocity
            velocity = v_token - self.time_max
            if velocity < 0 or velocity >= self.velocity_max:
                raise ValueError(f"Invalid velocity token: {v_token}")

            # Decode note
            note = n_token - self.time_max - self.velocity_max 
            if note <= 0 or note >= self.note_max:
                raise ValueError(f"Invalid note token: {n_token}")

            # Determine message type
     
            msg = Message('note_on', note=note, velocity=velocity, time=time_ms - prev_time_ms)
            prev_time_ms = time_ms

            messages.append((int(start_time_ms + time_ms), msg))

        return messages

if __name__ == "__main__":

    midi_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    wav_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.wav'

    import mido

    # Load the MIDI file
    midi_file = mido.MidiFile(midi_path)

    tok = Tokeniser()
    # times, msgs = tok.get_midi_between(midi_file, start_time=10000, end_time = 20000)
    times, msgs = tok.get_midi_between(midi_file, start_index=500, end_index = 1024)
    
    for t, m in zip(times[0:10], msgs[0:10]):
        print(t, m)
    messages = tok.detokenise_midi(tok.process_chunk(times, msgs), times[0])
    for m in messages[0:10]:
        print(m)
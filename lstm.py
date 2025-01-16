import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def train_network():
    notes = get_notes("music/*.mid")
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    train_model(model, network_input, network_output)

def get_notes(midi_dir):
    notes = []

    for file in glob.glob(midi_dir):
        print(f"Parsing {file}")
        midi = converter.parse(file)

        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse()
        except Exception:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    save_notes(notes, 'data/notes')
    return notes

def save_notes(notes, filepath):
    """ Save notes to a file """
    with open(filepath, 'wb') as file:
        pickle.dump(notes, file)

def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in sequence_in])
        network_output.append(note_to_int[sequence_out])

    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)

    return network_input, network_output

def create_network(network_input, n_vocab):
    model = Sequential([
        LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), recurrent_dropout=0.3, return_sequences=True),
        LSTM(512, recurrent_dropout=0.3, return_sequences=True),
        LSTM(512),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train_model(model, network_input, network_output):
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=[checkpoint])

if __name__ == '__main__':
    train_network()

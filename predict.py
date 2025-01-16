import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation

def generate():
    notes = load_notes('data/notes')

    pitchnames = sorted(set(notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input.shape, n_vocab, 'weights/weights.hdf5')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output, 'test_output.mid')

def load_notes(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def prepare_sequences(notes, pitchnames, n_vocab):
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    sequence_length = 100
    network_input = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])

    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    network_input = network_input / float(n_vocab)

    return network_input, network_input

def create_network(input_shape, n_vocab, weights_filepath):
    model = Sequential([
        LSTM(512, input_shape=(input_shape[1], input_shape[2]), recurrent_dropout=0.3, return_sequences=True),
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
    model.load_weights(weights_filepath)

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = {number: note for number, note in enumerate(pitchnames)}

    pattern = list(network_input[start])
    prediction_output = []

    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def create_midi(prediction_output, output_filepath):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes = [note.Note(int(n)) for n in pattern.split('.')]
            for n in notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filepath)

if __name__ == '__main__':
    generate()

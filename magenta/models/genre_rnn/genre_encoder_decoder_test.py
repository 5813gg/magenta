r"""Tests for the GenreEncoderDecoder.

To run this code:
$ bazel test genre_rnn:genre_encoder_decoder_test
"""

from magenta.lib import melodies_lib

from ...genre_rnn import genre_encoder_decoder

FLAGS = flags.FLAGS


class GenreEncoderDecoderTest(magic_test_thingy):

  def setUp(self):
    twinkle_twinkle = [14, 14, 21, 21, 23, 23, 21, 1]
    self.note_events = [e - 2 if e < 2 else e + 48 for e in twinkle_twinkle]
    self.num_genres = 2

  def testEncodeAsPop(self):
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(self.note_events)

    # test encoding as a pop sequence
    genre_id = 1
    pop_encoder = genre_encoder_decoder.GenreMelodyEncoderDecoder(
        genre_id, self.num_genres)
    pop_seq = pop_encoder.encode(melody)
    for feat in pop_seq.feature_lists.feature_list['inputs'].feature:
      bits = list(feat.float_list.value)
      self.assertTrue(sum(bits) == 2.0)
      self.assertTrue(bits[-1] == 1.0)
      self.assertTrue(bits[-2] == 0.0)
      break

  def testEncodeAsClassical(self):
    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(self.note_events)

    # test encoding as a classical sequence
    genre_id = 0
    classical_encoder = genre_encoder_decoder.GenreMelodyEncoderDecoder(
        genre_id, self.num_genres)
    classical_seq = classical_encoder.encode(melody)
    for feat in classical_seq.feature_lists.feature_list['inputs'].feature:
      bits = list(feat.float_list.value)
      self.assertTrue(sum(bits) == 2.0)
      self.assertTrue(bits[-1] == 0.0)
      self.assertTrue(bits[-2] == 1.0)
      break


if __name__ == '__main__':
  magic_test_thingy.main()

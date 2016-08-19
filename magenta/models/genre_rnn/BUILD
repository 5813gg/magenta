py_binary(
    name = "genre_classifier_train",
    srcs = ["genre_classifier_train.py"],
    deps = [
        ":genre_classifier",
        ":genre_ops",
        # tensorflow dep
    ],
)

py_binary(
    name = "genre_classifier",
    srcs = ["genre_classifier.py"],
    deps = [
        ":genre_ops",
        ":tensorflow_ops",
        "//magenta/lib:melodies_lib",
        "//magenta/lib:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "genre_ops",
    srcs = ["genre_ops.py"],
    deps = [
        ":genre_encoder_decoder",
        internal music_py_pb2",
        "magenta/lib:melodies_lib",
        "magenta/lib:sequences_lib",
        "magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_binary(
    name = "genre_encoder_decoder",
    srcs = ["genre_encoder_decoder.py"],
    deps = [
        "//magenta/lib:melodies_lib",
    ],
)

py_test(
    name = "genre_classifier_test",
    size = "medium",
    timeout = "long",
    srcs = ["genre_classifier_test.py"],
    data = [
        "testdata/test_genre_melodies.tfrecord",
    ],
    deps = [
        ":genre_classifier",
    ],
)

py_test(
    name = "genre_encoder_decoder_test",
    size = "medium",
    srcs = ["genre_encoder_decoder_test.py"],
    deps = [
        ":genre_encoder_decoder",
        "//magenta/lib:melodies_lib",
    ],
)

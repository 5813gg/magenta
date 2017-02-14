py_binary(
    name = "rl_tuner_train",
    srcs = ["rl_tuner_train.py"],
    deps = [
        ":rl_tuner",
        ":rl_tutor_ops",
        ":smiles_tutor",
        "//magenta/common:tf_lib",
    ],
)

py_binary(
    name = "rl_tuner",
    srcs = ["rl_tuner.py"],
    deps = [
        ":note_rnn_loader",
        ":rl_tutor_ops",
        ":rl_tutor",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "smiles_tutor",
    srcs = ["smiles_tutor.py"],
    deps = [
        ":smiles_rnn",
        ":rl_tutor_ops",
        ":rl_tutor",
        # tensorflow dep
    ],
)

py_binary(
    name = "rl_tutor",
    srcs = ["rl_tutor.py"],
    deps = [
        ":rl_tutor_ops",
        # tensorflow dep
    ],
)

py_binary(
    name = "note_rnn_loader",
    srcs = ["note_rnn_loader.py"],
    deps = [
        ":rl_tutor_ops",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        "//magenta/music:sequences_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "rl_tutor_ops",
    srcs = ["rl_tutor_ops.py"],
    deps = [
        "//magenta/common:tf_lib",
        # tensorflow dep
    ],
)

py_test(
    name = "rl_tuner_test",
    srcs = ["rl_tuner_test.py"],
    deps = [
        ":rl_tuner",
        ":rl_tuner_ops",
    ],
)

py_binary(
    name = "smiles_rnn_train",
    srcs = ["smiles_rnn_train.py"],
    deps = [
        ":smiles_rnn",
        "//magenta/common:tf_lib",
    ],
)

py_binary(
    name = "smiles_rnn",
    srcs = ["smiles_rnn.py"],
    deps = [
        ":rl_tutor_ops",
        # tensorflow dep
    ],
)
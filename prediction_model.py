def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=BATCH_SIZE,
    message_units=MESSAGE_UNITS,
    message_steps=MESSAGE_STEPS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    dense_units=DENSE_UNITS,
):

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(dense_units, activation="tanh")(x)

    x_XRD_1 = layers.Input((4400,1), dtype="float32", name="XRD")
    x_XRD_2 = layers.Convolution1D(filters=64,  kernel_size=2)(x_XRD_1)
    x_XRD_3 = layers.Flatten()(x_XRD_2)
    x_XRD_4 = layers.Dense(128, activation="tanh")(x_XRD_3)

    x_BET_1 = layers.Input((1), dtype="float32", name="BET")
    x_BET_2 = layers.Dense(1)(x_BET_1)

    concat = tf.keras.layers.Concatenate()([x, x_XRD_4, x_BET_2])

    x = layers.Dropout(0.3)(concat)
    x = layers.Dense(1, activation="relu")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator, x_XRD_1, x_BET_1],
        outputs=[x],
    )
    return model

mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)

mpnn.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
)

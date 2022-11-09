class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def radius(self, atom):
        atom_list = ['La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu','O']
        atom_radius = [1.36, 1.88, 1.61, 1.44, 1.37, 1.34, (0.58+0.53)/2, (0.55+0.585)/2, (0.545+0.53)/2, (0.56+0.48)/2, 0.65, 1.35]
        # from Supplementary Dataset 7 of https://doi.org/10.1038/s41467-020-17263-9
        # original reference: http://abulafia.mt.ic.ac.uk/shannon/radius.php
        return atom_radius[atom_list.index(atom.GetSymbol())]

    def electronegativity(self, atom):
        atom_list = ['La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu','O']
        atom_electronegativity = [1.1, 0.79, 0.89, 0.95, 1.13, 1, 1.55, 1.83, 1.88, 1.91, 1.9, 3.44]
        # from Supplementary Dataset 7 of https://doi.org/10.1038/s41467-020-17263-9  
        # original reference: https://en.wikipedia.org/wiki/Electronegativity
        return atom_electronegativity[atom_list.index(atom.GetSymbol())]

    def Asite(self, atom):
        atom_list = ['La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu','O']
        atom_Asite = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        return atom_Asite[atom_list.index(atom.GetSymbol())]

    def Bsite(self, atom):
        atom_list = ['La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu','O']
        atom_Bsite = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
        return atom_Bsite[atom_list.index(atom.GetSymbol())]

class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {'La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu','O'},
        "radius": {1.36, 1.88, 1.61, 1.44, 1.37, 1.34, (0.58+0.53)/2, (0.55+0.585)/2, (0.545+0.53)/2, (0.56+0.48)/2, 0.65, 1.35},
        "electronegativity": {1.1, 0.79, 0.89, 0.95, 1.13, 1, 1.55, 1.83, 1.88, 1.91, 1.9, 3.44},
        "Asite": {0, 1},
        "Bsite": {0, 1}
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

def molecule_from_pdb(pdb):
    molecule = Chem.rdmolfiles.MolFromPDBFile(pdb, sanitize=False)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

def graph_from_molecule(molecule):
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_list = ['La','Cs','Ba','Sr','Pr','Ca','Mn','Fe','Co','Ni','Cu','O']
        atom_radius = [1.36, 1.88, 1.61, 1.44, 1.37, 1.34, (0.58+0.53)/2, (0.55+0.585)/2, (0.545+0.53)/2, (0.56+0.48)/2, 0.65, 1.35]
        atom_electronegativity = [1.1, 0.79, 0.89, 0.95, 1.13, 1, 1.55, 1.83, 1.88, 1.91, 1.9, 3.44]
        atom_Asite = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        atom_Bsite = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
        symbol = atom_list.index(atom.GetSymbol())
        atom_features.append([atom_radius[symbol], atom_electronegativity[symbol], atom_Asite[symbol], atom_Bsite[symbol]])
        
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def graphs_from_pdb(pdb_list, seed_A):
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for pdb in pdb_list:
        molecule = molecule_from_pdb('/content/'+seed_A+'_'+pdb+'.pdb')
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )

def prepare_batch(x_batch, x_batch_XRD, x_batch_BET, y_batch):
    atom_features, bond_features, pair_indices = x_batch

    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    x_XRD, = x_batch_XRD
    x_XRD = x_XRD.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    x_BET, = x_batch_BET
    x_BET = x_BET.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator, x_XRD, x_BET), y_batch

def MPNNDataset(X, X_XRD, X_BET, y, batch_size=BATCH_SIZE, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, X_XRD, X_BET, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        for i in range(self.steps):
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)

class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=NUM_HEADS, embed_dim=EMBED_DIM, dense_dim=DENSE_DIM, batch_size=BATCH_SIZE, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)

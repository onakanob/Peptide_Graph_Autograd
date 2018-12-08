# Visualizing neural fingerprints.
#
# David Duvenaud
# Dougal Maclaurin
# 2015

import os, pickle
import autograd.numpy as np
#import autograd.numpy.random as npr
#from autograd import grad
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import matplotlib.pyplot as plt

from pepgraph import load_csv_test_split, relu
from pepgraph import build_conv_deep_net, build_convnet_fingerprint_fun
#from pepgraph import normalize_array, adam, build_batched_grad
from pepgraph import degrees, build_standard_net
#from pepgraph.util import rmse
from pepgraph.data_util import remove_duplicates

task_params = {'input_name': 'sequence',
               'target_name': 'Log_IC50',
               'conditions': {'mhc': 'HLA-A*02:01', 'peptide_length': '10'},
               'data_file': '../data/bdata.20130222.mhci.csv'}

num_epochs = 100
batch_size = 100
normalize = 1
dropout = 0
activation = relu
params = {'fp_length': 64,
            'fp_depth': 7,
            'init_scale':np.exp(-4),
            'learn_rate':np.exp(-4),
                    'b1':np.exp(-4),
                    'b2':np.exp(-4),
            'l2_penalty':np.exp(-4),
            'l1_penalty':np.exp(-5),
            'conv_width':32}

conv_layer_sizes = [params['conv_width']] * params['fp_depth']
conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                    'fp_length' : params['fp_length'],
                    'normalize' : normalize,
                    'return_atom_activations':False}

all_radii = range(params['fp_depth'] + 1)

# Plotting parameters
num_figs_per_fp = 4
figsize = (400, 400)
highlight_color = (30.0/255.0, 100.0/255.0, 255.0/255.0)  # A nice light blue.


def parse_training_params(params):
    nn_train_params = {'num_epochs'  : num_epochs,
                       'batch_size'  : batch_size,
                       'learn_rate'  : params['learn_rate'],
                       'b1'          : params['b1'],
                       'b2'          : params['b2'],
                       'param_scale' : params['init_scale']}

    vanilla_net_params = {'layer_sizes':[params['fp_length']],  # Linear regression.
                          'normalize':normalize,
                          'L2_reg': params['l2_penalty'],
                          'L1_reg': params['l1_penalty'],
                          'activation_function':activation}
    return nn_train_params, vanilla_net_params

def draw_molecule_with_highlights(filename, aa, highlight_atoms):
    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}   # Don't color nodes based on their element.
    drawoptions.bgColor=None

    mol = Chem.MolFromSequence(aa)
    fig = Draw.MolToMPL(mol, highlightAtoms=highlight_atoms, options=drawoptions,fitImage=True)

    fig.gca().set_axis_off()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def construct_atom_neighbor_list(array_rep):
    atom_neighbour_list = []
    for degree in degrees:
        atom_neighbour_list += [list(neighbours) for neighbours in array_rep[('atom_neighbors', degree)]]
    return atom_neighbour_list


def plot(trained_weights):
    print "Loading training data..."
    test_split = 0
    test_seed = 0
    train_data, _ = load_csv_test_split(\
        task_params['data_file'], test_split, test_seed,
        input_name=task_params['input_name'],
        target_name=task_params['target_name'],
        conditions=task_params['conditions'])
#    traindata, valdata, testdata = load_data(task_params['data_file'],
#                        (task_params['N_train'], task_params['N_valid'], task_params['N_test']),
#                        input_name='smiles', target_name=task_params['target_name'])
    train_aa, train_targets = train_data

    print "Convnet fingerprints with neural net"
    conv_arch_params['return_atom_activations'] = True
    output_layer_fun, parser, compute_atom_activations = \
       build_convnet_fingerprint_fun(**conv_arch_params)
    atom_activations, array_rep = compute_atom_activations(trained_weights, train_aa)

    if not os.path.exists('figures'): os.makedirs('figures')

    parent_molecule_dict = {}
    for mol_ix, atom_ixs in enumerate(array_rep['atom_list']):
        for atom_ix in atom_ixs:
            parent_molecule_dict[atom_ix] = mol_ix

    atom_neighbor_list = construct_atom_neighbor_list(array_rep)

    def get_neighborhood_ixs(array_rep, cur_atom_ix, radius):
        # Recursive function to get indices of all atoms in a certain radius.
        if radius == 0:
            return set([cur_atom_ix])
        else:
            cur_set = set([cur_atom_ix])
            for n_ix in atom_neighbor_list[cur_atom_ix]:
                cur_set.update(get_neighborhood_ixs(array_rep, n_ix, radius-1))
            return cur_set

    # Recreate trained network.
    nn_train_params, vanilla_net_params = parse_training_params(params)
    conv_arch_params['return_atom_activations'] = False
    _, _, combined_parser = \
        build_conv_deep_net(conv_arch_params, vanilla_net_params, params['l2_penalty'])

    net_loss_fun, net_pred_fun, net_parser = build_standard_net(**vanilla_net_params)
    net_weights = combined_parser.get(trained_weights, 'net weights')
    last_layer_weights = net_parser.get(net_weights, ('weights', 0))

    for fp_ix in range(params['fp_length']):
        print "FP {0} has linear regression coefficient {1}".format(fp_ix, last_layer_weights[fp_ix][0])
        combined_list = []
        for radius in all_radii:
            fp_activations = atom_activations[radius][:, fp_ix]
            combined_list += [(fp_activation, atom_ix, radius) for atom_ix, fp_activation in enumerate(fp_activations)]

        unique_list = remove_duplicates(combined_list, key_lambda=lambda x: x[0])
        combined_list = sorted(unique_list, key=lambda x: -x[0])

        for fig_ix in range(num_figs_per_fp):
            # Find the most-activating atoms for this fingerprint index, across all molecules and depths.
            activation, most_active_atom_ix, cur_radius = combined_list[fig_ix]
            most_activating_mol_ix = parent_molecule_dict[most_active_atom_ix]
            highlight_list_our_ixs = get_neighborhood_ixs(array_rep, most_active_atom_ix, cur_radius)
            highlight_list_rdkit = [array_rep['rdkit_ix'][our_ix] for our_ix in highlight_list_our_ixs]

            print "radius:", cur_radius, "atom list:", highlight_list_rdkit, "activation", activation
            draw_molecule_with_highlights(
                "figures/fp_{0}_pic_{1}_".format(fp_ix, fig_ix) + train_aa[most_activating_mol_ix] + ".pdf",
                train_aa[most_activating_mol_ix],
                highlight_atoms=highlight_list_rdkit)

if __name__ == '__main__':
    # Training.  Only need to run this part if we haven't yet saved results.pkl
#    trained_network_weights = train_neural_fingerprint()
#    with open('results.pkl', 'w') as f:
#        pickle.dump(trained_network_weights, f)

    # Plotting.
    with open('ExpC1.pkl', 'r') as f:
        trained_weights = pickle.load(f)
    plot(trained_weights)

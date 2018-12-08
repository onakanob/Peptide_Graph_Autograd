""" Testing peptide deep convnet
Based on regression example from https://github.com/HIPS/neural-fingerprint
Oliver Nakano-Baker"""

from time import time
import pickle
import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

from scipy.stats.stats import pearsonr

from pepgraph import load_csv_test_split
from pepgraph import build_conv_deep_net
from pepgraph import normalize_array, adam
from pepgraph import build_batched_grad
from pepgraph.util import rmse

from autograd import grad


def hyper_search(params, count):
    def exp_series(center, spread):
        return np.exp(npr.uniform(center-spread, center+spread, count))

    def lin_series(center, spread):
        return npr.uniform(center-spread, center+spread, count)

    def int_series(center, spread):
        return npr.randint(center-spread, high=center+spread+1, size=count)

    experiments = dict()
    for key in params:
        if params[key][2] == 'exp':
            experiments[key] = exp_series(params[key][0], params[key][1])
        elif params[key][2] == 'int':
            experiments[key] = int_series(params[key][0], params[key][1])
        else:
            experiments[key] = lin_series(params[key][0], params[key][1])

    return experiments


def plot_hypers(xlabel, x, ylabel, y, accuracy, title, save=False):
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.scatter(x, y, c=accuracy, cmap='Greens',
                edgecolors='k', linewidths=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.title(title)
    plt.show()
    if save:
        plt.savefig(title + '.png')

def plot_training(curve):
    plt.plot(range(0, 10*(len(curve[1])), 10),
             curve[1], label='training rmse')
    plt.plot(range(0, 10*(len(curve[2])), 10),
             curve[2], label='validation rmse')
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title(task_params['target_name'])
    plt.legend()
    plt.show()

def val_split(data, val_fraction, seed=np.array([])):
    if seed.any():
        npr.seed(seed)
    sequence = npr.choice(data[0].size, data[0].size, replace=False)
    val_lim = int(val_fraction * data[0].size)
    val_inputs = data[0][sequence[:val_lim]]
    val_targets = data[1][sequence[:val_lim]].astype('double')
    train_inputs = data[0][sequence[val_lim:]]
    train_targets = data[1][sequence[val_lim:]].astype('double')
    if seed.any():
        npr.seed()
    return train_inputs, train_targets, val_inputs, val_targets

def train_nn(pred_fun, loss_fun, num_weights, train_aa, train_raw_targets,
             train_params, seed=0,
             validation_aa=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 32
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = [[], [], []] # Test error, Val error
    def callback(weights, iter):
        if iter % 10 == 0:
            print "max of weights", np.max(np.abs(weights))
            selection = npr.choice(train_aa.size, size=num_print_examples)
            train_preds = undo_norm(pred_fun(weights, train_aa[selection]))
            cur_loss = loss_fun(weights, train_aa[selection], train_targets[selection])
#            train_preds = undo_norm(pred_fun(weights, train_aa[:num_print_examples]))
#            cur_loss = loss_fun(weights, train_aa[:num_print_examples], train_targets[:num_print_examples])
            training_curve[0].append(cur_loss)
            train_RMSE = rmse(train_preds, train_raw_targets[selection])
#            train_RMSE = rmse(train_preds, train_raw_targets[:num_print_examples])
            training_curve[1].append(train_RMSE)
            print "Iteration", iter, "loss", cur_loss,\
                  "train RMSE", train_RMSE,
            if validation_aa is not None:
                selection = npr.choice(validation_aa.size, size=num_print_examples)
                validation_preds = undo_norm(pred_fun(weights, validation_aa[selection]))
                val_RMSE = rmse(validation_preds, validation_raw_targets[selection])
                training_curve[2].append(val_RMSE)
                print "Validation RMSE", iter, ":", val_RMSE,

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_aa, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_aa):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_aa))
    return predict_func, trained_weights, training_curve

def run_experiment(train_inputs, train_targets, val_inputs, val_targets,
                   model_params, train_params, vanilla_net_params, filename=''):
    val_size = 1000
    conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
    conv_arch_params = {'num_hidden_features': conv_layer_sizes,
                        'fp_length': model_params['fp_length'],
                        'normalize': 1}
    loss_fun, pred_fun, conv_parser = \
        build_conv_deep_net(conv_arch_params, vanilla_net_params,
                            model_params['L2_reg'])
    num_weights = len(conv_parser)
    predict_func, trained_weights, conv_training_curve = \
        train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                 train_params, validation_aa=val_inputs,
                 validation_raw_targets=val_targets)

    if filename != '':
        with open(filename + '.pkl', 'w') as f:
            pickle.dump(trained_weights, f)

    train_selection = npr.choice(train_inputs.size, val_size)
    train_predictions = predict_func(train_inputs[train_selection])
    val_selection = npr.choice(val_inputs.size, val_size)
    val_predictions = predict_func(val_inputs[val_selection])
    plot_training(conv_training_curve)

    return predict_func,\
        pearsonr(train_predictions,
                 train_targets[train_selection])[0],\
        pearsonr(val_predictions,
                 val_targets[val_selection])[0]

# Experiment 1: choose hyper parameters for all sizes of HLA-A 2:1
task_params = {'input_name': 'sequence',
               'target_name': 'Log_IC50',
               'conditions': {'mhc': 'HLA-A*02:01',
                              'peptide_length': '10'},
               'data_file': '../data/bdata.20130222.mhci.csv'}
print 'Task: ', task_params
test_split = 0.2
test_seed = 0                   # Consistent seed to grab consistent test set
data, test_data = load_csv_test_split(
    task_params['data_file'], test_split, test_seed,
    input_name=task_params['input_name'],
    target_name=task_params['target_name'],
    conditions=task_params['conditions'])


hyper_params = dict(fp_length=(65, 4, 'int'),  # output pooling vector & FC layer input.
                    fp_depth=(7, 0, 'int'),  # The depth equals the fingerprint radius.
                    conv_width=(29, 5, 'int'),
                    h1_size=(127, 20, 'int'),  # Size of hidden layer of FC network.
                    L2_reg=(-2.125, 0.1, 'exp'),
                    num_iters=(64, 0, 'int'),
                    batch_size=(64, 0, 'int'),
                    init_scale=(-3.057, 0.5, 'exp'),
                    step_size=(-3.7, 0.2, 'exp'))
num_trials = 12
results = np.empty((2, num_trials))  # train; val
experiments = hyper_search(hyper_params, num_trials)

for i in range(num_trials):
    tic = time()
    train_inputs, train_targets, val_inputs, val_targets = val_split(data, 0.2)
    model_params = dict(fp_length=experiments['fp_length'][i],
                        fp_depth=experiments['fp_depth'][i],
                        conv_width=experiments['conv_width'][i],
                        h1_size=experiments['h1_size'][i],
                        L2_reg=experiments['L2_reg'][i])

    train_params = dict(num_iters=experiments['num_iters'][i],
                        batch_size=experiments['batch_size'][i],
                        init_scale=experiments['init_scale'][i],
                        step_size=experiments['step_size'][i])

    vanilla_net_params = dict(layer_sizes=[model_params['fp_length'],
                                           model_params['h1_size']],
                              normalize=True,
                              L2_reg=model_params['L2_reg'],
                              nll_func=rmse)
    try:
        (_, results[0, i], results[1, i]) =\
            run_experiment(train_inputs, train_targets,
                           val_inputs, val_targets,
                           model_params, train_params,
                           vanilla_net_params)  # train; val
        print 'Trial', i, '/', num_trials, 'took', (time()-tic)/60,\
            'minutes. Pearson:', results[1, i]
    except:
        print 'ERROR: Trial', i, 'failed to complete'

plot_hypers('fp_length', experiments['fp_length'],
            'conv_width', experiments['conv_width'],
            results[1, :], '')
plot_hypers('fp_length', experiments['fp_length'],
            'h1_size', experiments['h1_size'],
            results[1, :], '')
plot_hypers('init_scale', np.log(experiments['init_scale']),
            'step_size', np.log(experiments['step_size']),
            results[1, :], '')
plot_hypers('init_scale', np.log(experiments['init_scale']),
            'L2_reg', np.log(experiments['L2_reg']),
            results[1, :], '')

# Train on best results, test on test set
num_iters = 600
i = np.argmax(results[1, :])
model_params = dict(fp_length=experiments['fp_length'][i],
                    fp_depth=experiments['fp_depth'][i],
                    conv_width=experiments['conv_width'][i],
                    h1_size=experiments['h1_size'][i],
                    L2_reg=experiments['L2_reg'][i])

train_params = dict(num_iters=num_iters,
                    batch_size=experiments['batch_size'][i],
                    init_scale=experiments['init_scale'][i],
                    step_size=experiments['step_size'][i])

vanilla_net_params = dict(layer_sizes=[model_params['fp_length'],
                                       model_params['h1_size']],
                          normalize=True,
                          L2_reg=model_params['L2_reg'],
                          nll_func=rmse)

print "Best model:"
print model_params
print train_params
print vanilla_net_params

(predict_fun, results[0, i], results[1, i]) =\
    run_experiment(train_inputs, train_targets,
                   val_inputs, val_targets,
                   model_params, train_params,
                   vanilla_net_params, filename='ExpB')  # train; val
print 'Best model Pearson:', results[1, i]
(test_in, test_targets) = test_data
test_predictions = predict_fun(test_in)
test_correlation = pearsonr(test_predictions.astype('double'),
                            test_targets.astype('double'))
print 'Test Pearson correlation (logIC50):', test_correlation[0]

# Original data
test_correlation = pearsonr(np.exp(test_predictions.astype('double')),
                            np.exp(test_targets.astype('double')))
print 'Test Pearson correlation (IC50):', test_correlation[0]

# Try the network on 10-length peptides
task_params = {'input_name': 'sequence',
               'target_name': 'Log_IC50',
               'conditions': {'mhc': 'HLA-A*02:01',
                              'peptide_length': '9'},
               'data_file': '../data/bdata.20130222.mhci.csv'}
print 'Task: ', task_params
test_split = 0
test_seed = 0                   # Consistent seed to grab consistent test set
data_tens, _ = load_csv_test_split(\
    task_params['data_file'], test_split, test_seed,
    input_name=task_params['input_name'],
    target_name=task_params['target_name'],
    conditions=task_params['conditions'])

(test_in, test_targets) = data_tens
test_predictions = predict_fun(test_in)
test_correlation = pearsonr(test_predictions.astype('double'),
                            test_targets.astype('double'))
print 'Pearson correlation, 9-length peptides (logIC50):', test_correlation[0]

# Original data
test_correlation = pearsonr(np.exp(test_predictions.astype('double')),
                            np.exp(test_targets.astype('double')))
print 'Pearson correlation, 9-length peptides (IC50):', test_correlation[0]

#plot_mols(trained_weights)

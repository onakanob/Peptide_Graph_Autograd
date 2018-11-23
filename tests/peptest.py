""" Testing peptide deep convnet
Based on regression example from https://github.com/HIPS/neural-fingerprint
Oliver Nakano-Baker"""

import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

from pepgraph import load_data_csv

from pepgraph import build_conv_deep_net
from pepgraph import normalize_array, adam
from pepgraph import build_batched_grad
from pepgraph.util import rmse

from autograd import grad

task_params = {'input_name'  : 'sequence',
               'target_name' : 'Log_IC50',
               'conditions'  : {'mhc' : 'HLA-A*02:02'
#                                ,'peptide_length' : 9},
                               },
               'data_file'   : '../data/bdata.20130222.mhci.csv'}
#task_params = {'input_name'  : 'smiles',
#               'target_name' : 'measured log solubility in mols per litre',
#               'conditions'  : {},
#               'data_file'   : 'delaney.csv'}

# MHC HLA-A*02:02 all lengths:
N_train = 2400
N_val   = 830
N_test  = 830
               
# MHC HLA-A*02:02 only 9:
#N_train = 1450 #2400
#N_val   = 490 #830
#N_test  = 490 #830

# Delayney:
#N_train = 800
#N_val   = 80
#N_test  = 80
# TODO switch to percentage slicing
Pct_train = 0.6
Pct_val   = 0.2
Pct_test  = 0.2

model_params = dict(fp_length=50,    # Usually neural fps need far fewer dimensions than morgan.
                    fp_depth=4,      # The depth of the network equals the fingerprint radius.
                    conv_width=20,   # Only the neural fps need this parameter.
                    h1_size=100,     # Size of hidden layer of network on top of fps.
                    L2_reg=np.exp(-2))
train_params = dict(num_iters=100, #100
                    batch_size=100,
                    init_scale=np.exp(-4),
                    step_size=np.exp(-6))

# Define the architecture of the network that sits on top of the fingerprints.
vanilla_net_params = dict(
    layer_sizes = [model_params['fp_length'], model_params['h1_size']],  # One hidden layer.
    normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params, seed=0,
             validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 100
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = [[], [], []] # Test error, Val error
    def callback(weights, iter):
        if iter % 10 == 0:
            print "max of weights", np.max(np.abs(weights))
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve[0].append(cur_loss)
            train_RMSE = rmse(train_preds, train_raw_targets[:num_print_examples])
            training_curve[1].append(train_RMSE)
            print "Iteration", iter, "loss", cur_loss,\
                  "train RMSE", train_RMSE,
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                val_RMSE = rmse(validation_preds, validation_raw_targets)
                training_curve[2].append(val_RMSE)
                print "Validation RMSE", iter, ":", val_RMSE,

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve

def main():
    print "Loading data..."
    # Example Data:
    traindata, valdata, testdata = load_data_csv(
        task_params['data_file'], (N_train, N_val, N_test),
        # task_params['data_file'], (Pct_train, Pct_val, Pct_test), # TODO switch to percents
        input_name=task_params['input_name'], target_name=task_params['target_name'],
        conditions=task_params['conditions'])
    train_inputs, train_targets = traindata
    val_inputs,   val_targets   = valdata
    test_inputs,  test_targets  = testdata

    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", rmse(train_preds, train_targets)
        print "Test: ", rmse(val_preds,  val_targets)
        print "-" * 80
        return rmse(val_preds, val_targets)

    def run_conv_experiment():
        conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
        conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        loss_fun, pred_fun, conv_parser = \
            build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(conv_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)
#        x = range(10, 10*(1+len(conv_training_curve)), 10)
#        plt.plot(range(0, 10*(len(conv_training_curve[0])), 10),
#        conv_training_curve[0], label='training loss')
        plt.plot(range(0, 10*(len(conv_training_curve[1])), 10),
        conv_training_curve[1], label='training rmse')
        plt.plot(range(0, 10*(len(conv_training_curve[2])), 10),
        conv_training_curve[2], label='validation rmse')
        plt.xlabel('iteration')
        plt.ylabel('training loss')
        plt.title(task_params['target_name'])
        plt.legend()
        plt.show()
        test_predictions = predict_func(test_inputs)
        return rmse(test_predictions, test_targets)

    print "Task params", task_params
    print
    # print "Starting Morgan fingerprint experiment..."
    # test_loss_morgan = run_morgan_experiment()
    print "Starting neural fingerprint experiment..."
    test_loss_neural = run_conv_experiment()
    print "Neural test RMSE:", test_loss_neural
    # print "Morgan test RMSE:", test_loss_morgan, "Neural test RMSE:", test_loss_neural

if __name__ == '__main__':
    main()

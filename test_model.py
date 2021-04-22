import torch
from regression_model_and_dataset import BertForRegression
import my_utils as eutils
import sys
import argparse
from srblib import abs_path

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for getting the final test loss')

    ## Program Arguments
    parser.add_argument('--model_path', help='Model_Path)')
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int, help="kwarg passed to DataLoader")
    parser.add_argument('--pin_memory', action='store_true', help='Whether to pin memory or not. Default=False')
    parser.add_argument('--verbose', action='store_true',
                        help='Wthether to print the test loss in each iteration or not. Default = False')

    ## Parse the arguments
    args = parser.parse_args()
    assert args.model_path is not None, 'Please provide the fine-tuned model path'
    return args

def load_model(model_path, device, sanity_check=0):
    model = BertForRegression.load_from_checkpoint(model_path)
    if sanity_check: ## Sanity check to make sure model is fine-tuned i.e. model weights are different from original model
        aa = eutils.compare_models(model, BertForRegression())
        assert aa == 0, 'You are testing on the original model'
    model = model.to(device)
    return  model, model.tokenizer

def get_test_data_loader(model, batch_size=32, num_workers=0, pin_memory=False):
    if sys.gettrace() is not None: ## Debugging Mode
        pin_memory = False
        num_workers=0
    return model.get_dataloader('test', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def main(model_path, batch_sz, num_workers=0, pin_memory=False, verbose=True):
    ## Load the model
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(model_path, device=torch_device)

    ## Get DataLoader
    test_loader = get_test_data_loader(model, batch_size=batch_sz, num_workers=num_workers, pin_memory=pin_memory)

    ## Testing loop
    tot_loss = 0
    for l_idx, test_batch in enumerate(test_loader):
        ## Put data on device
        for key in test_batch.keys():
            test_batch[key] = test_batch[key].to(torch_device)

        ## Compute the loss
        loss = model._step(test_batch).item()
        tot_loss += loss

        if verbose:
            print(f'Type: Test, '
                  f'BatchIdx: {l_idx+1: 3d}/{len(test_loader)}, '
                  f'Per_Step_Test_Loss: {loss: 8.3f}')

    avg_loss = tot_loss/len(test_loader)
    print(f'Avergae test loss is {avg_loss}')

if __name__ == '__main__':
    # model_path = '/home/nzb0040/eluvio/lightning_logs/best_model_checkpoint-epoch=29-val_loss=290297.96875-step_count=0.ckpt'
    args = get_arguments()
    # import ipdb
    # ipdb.set_trace()
    main(model_path=abs_path(args.model_path), batch_sz=args.test_batch_size,
         num_workers=args.num_workers, pin_memory=args.pin_memory, verbose=args.verbose)
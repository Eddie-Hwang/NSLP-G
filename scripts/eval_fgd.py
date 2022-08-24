from argparse import ArgumentParser
from train_tfae import TransformerAutoencoderModule
import torch
from einops import rearrange


def joint_rearrange(x_inputs, device = 'cpu'):
    return [rearrange(x, 't v c -> t (v c)') for x in x_inputs]


def main(args):
    module = TransformerAutoencoderModule.load_from_checkpoint(args.ckpt)
    module.eval()
    module.freeze()

    model = module.model
    model = model.to(args.device)

    print(f'[INFO] Pretrained model and tokenizer are loaded from {args.ckpt}.')

    inputs = torch.load(args.input_dir)

    input_ids = inputs['ids']
    input_texts = inputs['texts']
    generated = inputs['outputs']
    reference = inputs['reference']

    if args.dataset_type == 'phoenix':
        assert len(generated) > 500, 'The number of sample is less than 500.'

    if len(generated[0].size()) == 3:
        generated, reference = map(lambda x: joint_rearrange(x, args.device), [generated, reference])
    
    # use all dataset samples
    score = model.eval_fgd(generated, reference, device = args.device)
    
    print('===================================')
    print(f'Dataset: {args.dataset_type}')
    print(f'Input dir: {args.input_dir}')
    print(f'FGD: {score:.3f}')
    print('===================================')

    
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_type', default = 'phoenix')
    parser.add_argument('--train_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.train')
    parser.add_argument('--valid_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.dev')
    parser.add_argument('--test_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.test')
    parser.add_argument('--ckpt', default = '/home/ejhwang/projects/NSLP-G/slp_logs/tfae/phoenix/checkpoints/last.ckpt')
    parser.add_argument('--input_dir', default = '/home/ejhwang/projects/aslp_pg/slp_logs/baseline/mean-phoenix-val.pt')
    parser.add_argument('--device', default = 'cuda:0')

    args = parser.parse_args()

    main(args)
    

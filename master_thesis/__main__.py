"""
Module containing the implementation of the CLI used to run the package.
"""
import argparse

import lpips
import pytorch_lightning as pl

import master_thesis as mt


def main(args):
    if args.chn and args.test:
        args.batch_size = 1
        args.frames_n = -1

    data = mt.MasterThesisData(**vars(args))
    data.prepare_data()

    trainer = pl.Trainer.from_argparse_args(args)

    model_vgg = mt.VGGFeatures.get_pretrained_model(
        'cuda:0' if args.gpus is not None else 'cpu'
    )

    if args.chn:
        model_lpips = lpips.LPIPS(net='alex')
        if args.chn_aligner == 'cpn':
            model_aligner = mt.CPN.init_model_with_state(
                args.chn_aligner_checkpoint
            )
        else:
            model_aligner = mt.DFPN.load_from_checkpoint(
                args.chn_aligner_checkpoint, model_vgg=model_vgg, **vars(args)
            )

        if args.test:
            model = mt.CHN.load_from_checkpoint(
                args.test_checkpoint, model_vgg=model_vgg,
                model_lpips=model_lpips, model_aligner=model_aligner,
                **vars(args)
            )
            trainer.test(model, data)
        else:
            model = mt.CHN(
                model_vgg=model_vgg, model_lpips=model_lpips,
                model_aligner=model_aligner, **vars(args)
            )
            trainer.fit(model, data)
    else:
        if args.test:
            model = mt.DFPN.load_from_checkpoint(
                args.test_checkpoint, model_vgg=model_vgg, **vars(args)
            )
            trainer.test(model, data)
        else:
            model = mt.DFPN(model_vgg, **vars(args))
            trainer.fit(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_checkpoint')
    parser.add_argument('--chn', action='store_true')
    parser.add_argument('--chn_aligner', choices=['dfpn', 'cpn'])
    parser.add_argument('--chn_aligner_checkpoint')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = mt.MasterThesisData.add_data_specific_args(parser)

    main(parser.parse_args())

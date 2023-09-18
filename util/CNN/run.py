#!/usr/bin/env python3

import argparse
import sys
import os
import LSTM

def main(args):
    epochs = 1000
    U_path = None
    V_path = None
    target_path = None
    learning_rate = 1e-5
    batchsize = 1
    device = "cuda"
    modelpath = None
    if args.subcommand == "epochs":
        temp = args.epochs
        epochs = int(temp)
    if args.subcommand == "U_path":
        U_path = args.U_path
    if args.subcommand == "V_path":
        V_path = args.V_path
    if args.subcommand == "target_path":
        target_path = args.target_path
    if args.subcommand == "learning_rate":
        temp = args.learning_rate
        learning_rate = float(temp)
    if args.subcommand == "batchsize":
        temp = args.batchsize
        batchsize = int(temp)
    if args.subcommand == "device":
        device = args.device


    LSTM.train(U_path = U_path, V_path = V_path, target_path = target_path, num_output = 9,
          learning_rate = learning_rate, batchsize= batchsize,  device = device, epochs = epochs, modelpath = modelpath)

    sys.exit(0)



def setup_parser():
    """Setup argument parser and subparsers."""
    parser = argparse.ArgumentParser(description="Simple CLI Template.")

    subparsers = parser.add_subparsers(dest="subcommand", help="Sub-command help.")

    epochs_parser = subparsers.add_parser("epochs", help="epochs help")
    epochs_parser.add_argument("epochs", help="epochs")

    U_path_parser = subparsers.add_parser("U_path", help="U_path help")
    U_path_parser.add_argument("U_path", help="U_path")

    V_path_parser = subparsers.add_parser("V_path", help="V_path help")
    V_path_parser.add_argument("V_path", help="V_path")

    target_path_parser = subparsers.add_parser("target_path", help="target_path help")
    target_path_parser.add_argument("target_path", help="target_path")

    learning_rate_parser = subparsers.add_parser("learning_rate", help="learning_rate help")
    learning_rate_parser.add_argument("learning_rate", help="learning_rate")

    batchsize_parser = subparsers.add_parser("batchsize", help="batchsize help")
    batchsize_parser.add_argument("batchsize", help="batchsize")

    device_parser = subparsers.add_parser("device", help="device help")
    device_parser.add_argument("device", help="device")



    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()


    main(args)

import argparse

import utils.gpt_token_encoder as encoder
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size.')
    parser.add_argument('--input_seq_len', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--o_file_name', type=str, default='./sample_input.txt',
                        help='path to the sample text file.')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')

    args = parser.parse_args()

    batch_size = args.batch_size
    input_seq_len = args.input_seq_len
    o_file_name = args.o_file_name

    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print ("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

    with open( "../sample/source_sentence.txt", "r" ) as fp :
        src_sentence = fp.read()

    enc = encoder.get_encoder(args.vocab_file, args.merges_file)

    contexts = [ src_sentence ]
    start_ids = [torch.IntTensor( enc.encode(c)) for c in contexts]
    start_ids = np.array( start_ids[0] )

    start_ids = start_ids[:input_seq_len]

    output = enc.decode( start_ids )

    with open( o_file_name, "w" ) as fp :
        for _ in range( batch_size ) :
            fp.write( output + '\n' )

    return

if __name__ == '__main__':
    main()


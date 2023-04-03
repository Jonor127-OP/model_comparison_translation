import argparse

def merge_src_tgt(src, tgt, output_file):

    with open(src, 'r') as src_file:
        src_file = src_file.readlines()

    with open(tgt, 'r') as tgt_file:
        tgt_file = tgt_file.readlines()

    merged_lines = zip(src_file, tgt_file)

    with open(output_file, 'w') as merged_file:
        for line1, line2 in merged_lines:
            merged_file.write(line1.strip() + ' <sep> ' + line2)


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("src_file", type=str, help="The source file")
    parser.add_argument("tgt_file", type=str, help="The target file")
    parser.add_argument("output_file", type=str, help="The output file")
    args = parser.parse_args()

    # Call the transform_to_ids function with the command-line arguments
    merge_src_tgt(args.src_file, args.tgt_file, args.output_file)
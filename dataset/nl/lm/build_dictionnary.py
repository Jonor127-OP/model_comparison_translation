#!/usr/bin/env python3

from collections import OrderedDict
import fileinput
import sys

import numpy
import json


def main():
    i = 4
    worddict = OrderedDict()
    worddict['<eos>'] = 1
    worddict['<sos>'] = 2
    worddict['<unk>'] = 3
    worddict['<pad>'] = 0
    worddict['<sep>'] = 5

    for filename in sys.argv[1:]:
        print('Processing', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in worddict:
                        worddict[w] = i
                        i += 1

        # The JSON RFC requires that JSON text be represented using either
        # UTF-8, UTF-16, or UTF-32, with UTF-8 being recommended.
        # We use UTF-8 regardless of the user's locale settings.
        with open('%s.json'%filename, 'w', encoding='utf-8') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        print('Done')


if __name__ == '__main__':
    main()
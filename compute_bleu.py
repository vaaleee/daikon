#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imported modules
import argparse # for argument handling
import codecs # for opening files
from nltk import ngrams # for creating ngrams
from nltk.tokenize import word_tokenize # for tokenization
import numpy as np # for exp function
from collections import Counter # for counting ngrams
import typing

################################################################################

def parse_args() -> argparse.Namespace:
    """ Parse arguments given via command lines

    Keyword arguments:  None

    Returns:   args - object of class argparse.Namespace
    """

    parser = argparse.ArgumentParser()

    # required arguments: a target file, the source file translated into the target language
    parser.add_argument("-trg", "--target",  required=True, action="store", dest="trg", help="target text")
    parser.add_argument("-trans", "--translation",  required=True, action="store", dest="trans", help="source translation")

    args = parser.parse_args()

    return args

################################################################################

def tokenize_file(filename: str) -> list:
    """ Open a file, read and tokenize content

    Keyword arguments:  filename - string, name of a file

    Returns:    text - list, tokenized text of file
    """

    with codecs.open(filename,"r","utf-8") as infile:

        # read content and tokenize text
        text = tokenize(infile.read())

    return text

################################################################################

def tokenize(sentence: str) -> list:
    """ Tokenize a given sentence

    Keyword arguments:  sentence - string, a sentence

    Returns:    sentence - list of tokens
    """

    # tokenize sentence with nltk tokenizer
    tokenized = word_tokenize(sentence)
    return tokenized

################################################################################

def compute_ngram_precision(hypothesis:list, reference:list, n: int) -> float:
    """ Compute the ngram precision for a given sentence and ngram size

    Keyword arguments:  hypothesis - list of tokens, a translated sentence
                        reference - list of tokens, reference translation of sentences
                        n - integer, number of n-grams considered

    Returns:    precision - float, ngram precision for given sentence and ngram size
    """

    # base case; if all ngram sizes processed, stop recursion
    if n == 0:
        return 1

    # create ngrams of size n, initialise frequencey dictionary for clipping
    ngrams_ref = ngrams(reference,n)
    ngrams_sent = ngrams(hypothesis,n)
    clip_dictionary = Counter(ngrams_ref)

    # initialise counters
    total = 0.
    correct = 0.

    # check how many ngrams in the sentence also occur in the reference (includes clipping)
    for ngram in ngrams_sent:
        total += 1
        if clip_dictionary[ngram] > 0:
            correct += 1
            clip_dictionary[ngram] -= 1

    # compute the precision
    precision =  correct/total

    # recursively call function for all ngram sizes
    return precision * compute_ngram_precision(hypothesis,reference,n-1)

################################################################################

def compute_bleu_score(hypothesis:list, reference:list, n:int=4) -> float:
    """ Compute the BLEU score for a given sentence

    Keyword arguments:  hypothesis - list of tokens, a translated sentence
                        reference - list of tokens, reference translation of sentences
                        n - integer, number of n-grams considered (default = 4)

    Returns:    score - float, BLEU score for given sentence
    """

    # compute brevity penalty
    bp = min(1.0,np.exp(1 - len(reference)/len(hypothesis)))

    # compute geometric mean of ngram precisions
    p = compute_ngram_precision(hypothesis,reference,n)**(1/n)

    # compute final bleu score
    score = bp * p

    return score


################################################################################

def main(args: argparse.Namespace) -> None:
    """ Main function to compute BLEU score of a given file and a reference translation

    Keyword arguments:  args - object of class argparse.Namespace

    Returns:    None
    """

    # read target and translation files and tokenize text
    trg = tokenize_file(args.trg)
    trans = tokenize_file(args.trans)

    # compute bleu score
    bleu_score = compute_bleu_score(trans,trg)

    # print information
    print("BLEU score for {0:s}:".format(args.trans))
    print("\t",bleu_score)

################################################################################

if __name__ == "__main__":

    args = parse_args()
    main(args)

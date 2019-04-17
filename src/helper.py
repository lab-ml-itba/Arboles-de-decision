import numpy as np
import matplotlib.pyplot as plt

def entropy(dist):
    out=0
    for key,value in dist.items():
        out -= value*np.log2(value)
    return out
    
def combine(dist1,dist2):
    out=dict()
    for key1,value1 in dist1.items():
        for key2,value2 in dist2.items():
            out[key1+key2]=np.around(value1*value2,decimals=10)
    return out

def huffman(p):
    '''Return a Huffman code for an ensemble with distribution p.'''
    #assert(sum(p.values()) == 1.0) # Ensure probabilities sum to 1

    # Base case of only two symbols, assign 0 or 1 arbitrarily
    if(len(p) == 2):
        return dict(zip(p.keys(), ['0', '1']))

    # Create a new distribution by merging lowest prob. pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = huffman(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + '0', ca1a2 + '1'

    return c

def lowest_prob_pair(p):
    '''Return pair of symbols from distribution p with lowest probabilities.'''
    assert(len(p) >= 2) # Ensure there are at least 2 symbols in the dist.

    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]
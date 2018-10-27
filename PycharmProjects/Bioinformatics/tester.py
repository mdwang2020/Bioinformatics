import numpy as np
import sys
from ncpol2sdpa import *
from scipy.misc import *


def function(D, m_deg, nk, x):
    """
    defines the cost function
    :param m_deg: the highest moment
    :param nk: n_A + n_C + n_G + n_T
    :param x: alpha
    :return: returns the cost function, represented by symbols
    """
    n = sum(nk)
    sum_2 = 0
    sum_1 = 0
    for k in range(0, 4):
        sum_1 = 0
        for t in range(1, m_deg):
            sum_1 = n ** (m_deg - t) * comb(m_deg, t - 1) * nk[k] ** (m_deg - 1) * \
                    np.dot(x, D[k][m_deg - t + 1]) * (np.dot(x, D[k][1])) ** (t - 1)
        sum_2 += nk[k] * (np.dot(x, D[k][1]) ** m_deg)
    return sum_1 + n * sum_2


def bounds(n_var, x):
    """
    sets the inequalities for alpha. 0 < alpha < 1
    :param x: alpha
    :param n: number of variables
    :return: a list of bound constraints
    """
    inequalities = list()
    for i in range(n_var):
        inequalities.append(x[i])
        inequalities.append(1 - x[i])
    return inequalities


def equalities(D, m_deg, nk, x):
    """
    defines the constraints of the problem
    :param m_deg: the highest moment
    :param nk: n_A + n_C + n_G + n_T
    :param x: alpha
    :return: returns the constraints of the cost function in symbols
    """
    n = sum(nk)
    equalities = list()
    sum_1 = 0
    for i in range(N):
        sum_1 += x[i]
    equalities.append(sum_1 - 1)
    for k in range(4):
        equalities.append(np.dot(x, D[k][0]) - nk[k])
    for i in range(1, m_deg):
        equalities.append(function(D, i, nk, x) - exp_sum(i, n))
    return equalities


def save_natural_vector(sequence_file, vector_file, m):
    """
    Converts the sequence to a natural vector
    :param sequence_file: DNA sequence
    :param vector_file: file where the converted vector is saved to
    :param m: degree of the moment
    :return: "vector file saved"
    """
    vec_file = open(vector_file, 'w')
    with open(sequence_file) as fp:
        line = fp.readline()
        line_counter = 1
        while line:
            if line_counter % 2 == 0 and len(line.strip()) > 1:
                vect = converter_m(line.strip(), m)
                vec_file.write(','.join(str(e) for e in vect))
                vec_file.write('\n')
            line = fp.readline()
            line_counter += 1
    return 'vector file saved'


def converter_m(sequence, m):
    """
    Convert DNA sequence to natural vector of m-th moments
    :param sequence: 'accgttacct'
    :param m: the order of the highest moment
    :return: list that represents the natural vector of dimension 4 * (m + 1)
    """
    na_vect = [0] * (4 * (m + 1))
    counter = [0] * 4
    pos_sum = [0] * 4
    # count number of appearance
    for i in range(0, len(sequence)):
        counter[index_map(sequence[i])] += 1
        pos_sum[index_map(sequence[i])] += i
    # populate n_k and mu_k
    for k in range(0,4):
        na_vect[k] = counter[k]
        na_vect[k + 4] = pos_sum[k] / counter[k]
    n = sum(counter)
    for i in range(0, len(sequence)):
        for t in range(2, m+1):
            for k in range(0,4):
                na_vect[4*t+k] += (i - na_vect[k+4])**t / n**(t-1) / na_vect[k]**(t-1)
    return na_vect


def index_map(letter):
    """
    Defines the index of each letter in the sequence
    :param letter: an a, c, t, or g
    :return: 0
    """
    if letter == 'a':
        return 0
    elif letter == 'c':
        return 1
    elif letter == 'g':
        return 2
    elif letter == 't':
        return 3
    else:
        return 0


def nv_converter(sequence_file):
    """
    represents nk and mu_k as D[k][0] and D[k][1], respectively
    :param sequence_file: the file where the result goes into
    :return: D is a matrix of 4 x m+1 x N dimensions, N is the number of base sequences
    """
    D = [[[] for j in range(m + 1)] for k in range(4)]
    N = 0
    """
        D[k][0] is nk, where nk is the sum of n_A, n_C, n_G, and n_T
        D[k][1] is mu_k. mean of the positions
        D[k][n] n >=2. represents the nth moment
    """
    sf = open(sequence_file)
    line = sf.readline()
    while line:
        line = line.split(",")
        if len(line) > 1:
            for t in range(m + 1):
                D[0][t].append(float(line[t * 4]))
                D[1][t].append(float(line[t * 4 + 1]))
                D[2][t].append(float(line[t * 4 + 2]))
                D[3][t].append(float(line[t * 4 + 3]))
            N += 1
        line = sf.readline()
    return D, N


def exp_sum(m, n):
    """
    Defines the summation of the exponent of the index.
    :param n: number of summation items
    :param m: an integer power of the exponent
    :return: a whole number
    """
    tsm = 0
    for i in range(1, n + 1):
        tsm += i ** m
    return tsm


if __name__ == "__main__":
    # the number of As, Cs, Ts, and Gs. To be tested
    test_nk = [2890, 1436, 1817, 1875]
    level = 2  # Requested level of relaxation, for accuracy
    # enter in terminal "python tester.py m", where m >= 2
    m = int(sys.argv[1])
    nk = test_nk
    n = sum(test_nk)
    # generate natural vector file
    save_natural_vector('group_M_shortest10.fasta', 'vectors_1.txt', m)
    sequence_file = 'C:\\Users\\pizza\\PycharmProjects\\Bioinformatics\\vectors_1.txt'
    D, N = nv_converter(sequence_file)
    """
        D[k][0] is nk, where nk is the sum of n_A, n_C, n_G, and n_T
        D[k][1] is mu_k. mean of the positions
        D[k][n] n >=2. represents the nth moment
        N is the number of sequences
    """
    n_vars = N  # Number of variables
    x = generate_variables('x', n_vars)
    m_deg = m
    obj = function(D, m_deg, nk, x)
    print(obj)
    sdp = SdpRelaxation(x)
    inequalities = bounds(n_vars, x)
    equalities = equalities(D, m_deg, nk, x)
    sdp.get_relaxation(level, objective=obj, inequalities=inequalities, equalities=equalities)
    sdp.solve()
    # The results from the optimizer. There is a solution to the problem only when the status is "optimal"
    print(sdp.primal, sdp.dual, sdp.status)
    # Prints the points of optimization
    for i in range(n_vars):
        print((sdp[x[i]]))
    # Evaluates the cost function at the minimized point
    subs = {}
    for i in range(n_vars):
        subs[x[i]] = sdp[x[i]]
    obj.evalf(subs=subs)
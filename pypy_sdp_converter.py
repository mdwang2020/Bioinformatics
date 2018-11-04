import numpy as np
import sys
from ncpol2sdpa import *


def comb(n, k):
    """
    a function for n choose k. A separate function was necessary because pypy does not support scipy, which comb was a
    function for.
    :param n: a number
    :param k: a number
    :return: a number
    """
    if n < k or n < 0 or k < 0:
        raise ValueError('One of the numbers is < 0 or n < k')
    return np.math.factorial(n) / np.math.factorial(k) / np.math.factorial(n - k)


def sym_pow(sym_exp, power):
    """
    a function that replaces the ** operator for a more optimal performance
    :param sym_exp: a variable expression
    :param power: a non-negative number or variable expression
    :return: an expression
    """
    if power < 0:
        raise ValueError('Power Error: power < 0')
    pow_exp = 1
    for i in range(power):
        pow_exp *= sym_exp
    return pow_exp


def s_func(D, m, nk, x):
    """
    defines the cost function
    :param D: data from vector file
    :param m: the highest moment
    :param nk: n_A + n_C + n_G + n_T
    :param x: alpha
    :return: returns the cost function, represented by symbols
    """
    """
        D[k][0] is nk, where nk is the sum of n_A, n_C, n_G, and n_T
        D[k][1] is mu_k. mean of the positions
        D[k][n] n >=2. represents the nth moment
        N is the number of sequences
    """
    n = sum(nk)
    sum_2 = 0
    sum_1 = 0
    for k in range(0, 4):
        sum_1 = 0
        print('k=%d' % k)
        for t in range(1, m):
            print('t=%d' % t)
            # need to convert n ** (m - t) * comb(m, t - 1) * nk[k] ** (m - 1) from long to float, otherwise error
            # use sym_pow function to speed up
            sum_1 = float(n ** (m - t) * comb(m, t - 1) * nk[k] ** (m - t)) * \
                    np.dot(x, D[k][m - t + 1]) * sym_pow(np.dot(x, D[k][1]), t - 1)
        sum_2 += nk[k] * sym_pow(np.dot(x, D[k][1]), m)
    return sum_1 + sum_2


def bounds(n, x):
    """
    sets the inequalities for alpha. 0 < alpha < 1
    :param x: alpha
    :param n: number of variables
    :return: a list of bound constraints
    """
    bound_conditions = list()
    for i in range(n):
        bound_conditions.append(x[i])
        bound_conditions.append(1 - x[i])
    return bound_conditions


def equalities_func(D, m, nk, x):
    """
    defines the constraints of the problem
    :param m: the highest moment
    :param D:
    :param nk: n_A + n_C + n_G + n_T
    :param x: alpha
    :return: returns the constraints of the cost function in symbols
    """
    n = sum(nk)
    equality_list = list()
    sum_1 = 0
    for i in range(N):
        sum_1 += x[i]
        equality_list.append(sum_1 - 1)
    for k in range(4):
        equality_list.append(np.dot(x, D[k][0]) - nk[k])
    for i in range(1, m):
        equality_list.append(s_func(D, i, nk, x) - exp_sum(i, n))
    return equality_list


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
            #print(line)
            line_counter += 1
    vec_file.close()
    return 'vector file %s was saved' % vector_file


def converter_m(sequence, m):
    """
    Convert DNA sequence to natural vector of m-th moments
    :param sequence: 'accgttacct'
    :param m: the order of the highest moment
    :return: list that represents the natural vector of dimension 4 * (m + 1)
    """
    #print(sequence)
    na_vect = [0] * (4 * (m + 1))
    counter = [0] * 4
    pos_sum = [0] * 4
    # count number of appearance
    for i in range(0, len(sequence)):
        counter[index_map(sequence[i])] += 1
        pos_sum[index_map(sequence[i])] += i
    # populate n_k and mu_k
    for k in range(0, 4):
        na_vect[k] = counter[k]
        na_vect[k + 4] = pos_sum[k] / counter[k]
    n = sum(counter)
    for i in range(0, len(sequence)):
        for t in range(2, m + 1):
            for k in range(0, 4):
                na_vect[4 * t + k] += (i - na_vect[k + 4]) ** t / n ** (t - 1) / na_vect[k] ** (t - 1)
    return na_vect


def index_map(letter):
    """
    Defines the index of each letter in the sequence
    :param letter: an a, c, t, or g
    :return: 0
    """
    low_letter = letter.lower()
    if low_letter == 'a':
        return 0
    elif low_letter == 'c':
        return 1
    elif low_letter == 'g':
        return 2
    elif low_letter == 't':
        return 3
    else:
        return 0


def nv_converter(vector_file, m):
    """
    represents nk and mu_k as D[k][0] and D[k][1], respectively
    :param sequence_file: the file where the result goes into
    :return: D is a matrix of 4 x m + 1 x N dimensions, N is the number of base sequences
    """
    D = [[[] for j in range(m + 1)] for k in range(4)]
    N = 0
    """
        D[k][0] is nk, where nk is the sum of n_A, n_C, n_G, and n_T
        D[k][1] is mu_k. mean of the positions
        D[k][n] n >=2. represents the nth moment
    """
    print('read vector file ...%s' % vector_file)
    with open(vector_file) as sf:
        for line in sf:
            #print(line)
            line = line.split(",")
            if len(line) > 1:
                for t in range(m + 1):
                    D[0][t].append(float(line[t * 4]))
                    D[1][t].append(float(line[t * 4 + 1]))
                    D[2][t].append(float(line[t * 4 + 2]))
                    D[3][t].append(float(line[t * 4 + 3]))
                N += 1
    
    return D, N


def exp_sum(m, n):
    """
    Defines the summation of the exponent of the index.
    :param m: an integer power of the exponent
    :param n: number of summation items
    :return: a whole number
    """
    tsm = 0
    for i in range(1, n + 1):
        tsm += i ** m
    return tsm


if __name__ == "__main__":
    # enter in terminal "python tester.py m", where m >= 2
    max_degree = int(sys.argv[1])

    # the number of As, Cs, Ts, and Gs. To be tested
    test_nk = [2890, 1436, 1817, 1875]
    level = 2  # Requested level of relaxation, for accuracy
    min_model_file = 'seq_ver_min_model.dat-s'
    max_model_file = 'seq_ver_max_model.dat-s'
    sequence_file = 'group_M_shortest20.fasta'
    vector_file = 'SeqVer_vectors.txt'

    # generate natural vector file
    #save_natural_vector('SeqVer.fasta', 'SeqVer_vectors.txt', max_degree)
    save_natural_vector(sequence_file, vector_file, max_degree) 
    D, N = nv_converter(vector_file, max_degree)
    print('natural vector file saved: %s' % sequence_file)
    """
        D[k][0] is nk, where nk is the sum of n_A, n_C, n_G, and n_T
        D[k][1] is mu_k. mean of the positions
        D[k][n] n >=2. represents the nth moment
        N is the number of sequences
    """
    print('Total number of natural vector: %d' % N)
    n_vars = N  # Number of variables
    x = generate_variables('x', n_vars)

    #print(obj)
    inequalities = bounds(n_vars, x)
    equalities = equalities_func(D, max_degree, test_nk, x)

    sdp = SdpRelaxation(x)

    # Find the minimum
    print('find min start...')
    obj = s_func(D, max_degree, test_nk, x)
    print('Starting sdp conversion ...')
    sdp.get_relaxation(level, objective=obj, inequalities=inequalities, equalities=equalities)
    print('sdp relaxation finished')
    print('saving SDPA model file ...')
    sdp.write_to_file(min_model_file)
    print('SDPA model file saved: %s' % min_model_file)

    """
    # The solver only works when there's an optimizer installed, such as MOSEK.
    sdp.solve()
    # The results from the optimizer. There is a solution to the problem only when the status is "optimal"
    print(sdp.primal, sdp.dual, sdp.status)
    if sdp.status == "optimal":
        min_cost = sdp.primal
    else:
        min_cost = 0
    # Prints the points of optimization
    for i in range(n_vars):
        print((sdp[x[i]]))
    # Evaluates the cost function at the minimized point
    subs = {}
    for i in range(n_vars):
        subs[x[i]] = sdp[x[i]]
    obj.evalf(subs=subs)
    """

    # Find the maximum 
    print('find max start...')
    obj = -s_func(D, max_degree, test_nk, x)
    print('Starting sdp conversion ...')
    sdp.get_relaxation(level, objective=obj, inequalities=inequalities, equalities=equalities)
    print('sdp relaxation finished')
    print('saving SDPA model file ...')
    sdp.write_to_file(max_model_file)
    print('SDPA model file saved: %s' % max_model_file)

    """
    # The solver only works when there's an optimizer installed, such as MOSEK.
    sdp.solve()
    # The results from the optimizer. There is a solution to the problem only when the status is "optimal"
    print(sdp.primal, sdp.dual, sdp.status)
    if sdp.status == "optimal":
        max_cost = -sdp.primal
    else:
        max_cost = 0
    # Prints the points of optimization
    for i in range(n_vars):
        print((sdp[x[i]]))
    # Evaluates the cost function at the minimized point
    subs = {}
    for i in range(n_vars):
        subs[x[i]] = sdp[x[i]]
    obj.evalf(subs=subs)

    # Compares the minimum and the maximum to the exponential summation
    if min_cost <= exp_sum(max_degree, sum(test_nk)) <= max_cost:
        print('There is a solution.')
    else:
        print('There is not a solution.')
    """

from scipy.misc import *
import numpy as np
from scipy.optimize import *
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
import sys

#Global variables are necessary here because the nonlinear constraint in the solver itself are unable to take in args
D = None
N = None
m_deg = None
nk = None


def save_natural_vector(sequence_file, vector_file, m):
    """
    Converts the sequence to a natural vector
    :param sequence_file: DNA sequence
    :param vector_file: file where the converted vetor is saved to
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


def linear_constraint_matrix():
    """
    Defines a linear constraint matrix, which consists of the constraint of alpha, the n_k constraints, and mu_k
    constraints.
    :return: 6 x N matrix
    """
    #D, N, and nk are global variables defined at the top of the code
    c = [[], [], [], [], [], []]
    for i in range(0, N):
        c[0].append(1)
    for k in range(0, 4):
        for i in range(0, N):
            c[k + 1].append(D[k][0][i])
    for i in range(0, N):
        e = 0
        for k in range(0, 4):
            e += nk[k] * D[k][1][i]
        c[5].append(e)
    return c


def non_linear_const_matrix(x):
    """
    Defines the nonlinear constraint list.
    :param x: alpha
    :return: a list with m_deg - 2 terms
    """
    #D, N, m_deg, nk = args[0], args[1], args[2], args[3]
    # The four variables above are used instead by the global variables defined at the top of the code
    n = sum(nk)
    sum_2 = 0
    sum_1 = 0
    result_list = []
    for m_ix in range(2, m_deg):
        for k in range(0, 4):
            sum_1 = 0
            for t in range(1, m_ix):
                sum_1 = n ** (m_ix - t) * comb(m_ix, t - 1) * nk[k] ** (m_ix - 1) * \
                        np.dot(x, D[k][m_ix - t + 1]) * (np.dot(x, D[k][1])) ** (t - 1)
            sum_2 += nk[k] * (np.dot(x, D[k][1]) ** m_ix)
        result_list.append(sum_1 + sum_2)
    return result_list


def exp_sum(n, m):
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


def linear_bound(nk):
    """
    Defines the lower and upper bounds for linear constraints
    :param nk: a 4 element list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
               projection)
    :param m: the highest order moment
    :return: a list with 6 terms
    """
    n = sum(nk)
    bound = [1, nk[0], nk[1], nk[2], nk[3], exp_sum(n, 1)]
    return bound


def nonlinear_bound(nk, m):
    """
    Defines the lower and upper bounds for nonlinear constraints. m >= 3.
    :param nk: a 4 element list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
               projection)
    :param m: the highest order moment
    :return: a list with m - 2 terms
    """
    n = sum(nk)
    result_list = []
    for i in range(2, m):
        result_list.append(exp_sum(n, i))
    return result_list


def nonlinear_jac_matrix(x):
    """
        Defines the nonlinear constraint jacobian matrix
        :param x: alpha
        :return: a matrix with dimensions N x (m - 2)
        """
    #D, N, m_deg, nk = args[0], args[1], args[2], args[3]
    #The four variables above are used instead by the global variables defined at the top of the code
    result_matrix = []
    for i in range(2, m_deg):
        args_local = (D, N, i, nk)
        result_matrix.append(f_jac(x, *args_local))
    return result_matrix


def nonlinear_hess_matrix(x, v):
    """
        Defines the linear combinations of the nonlinear constraint hessian matrices
        :param x: alpha
        :param v: An auxillary variable used by the optimizer
        :return: N matrices with dimensions N x (m - 2)
        """
    #D, N, m_deg, nk = args[0], args[1], args[2], args[3]
    # The four variables above are used instead by the global variables defined at the top of the code
    sum_1 = 0
    for i in range(2, m_deg):
        args_local = (D, N, i, nk)
        sum_1 += v[i - 2] * np.array(f_hess(x, *args_local))
    return sum_1


def f(x, *args):
    '''
    Defines the cost function
    :param x: alpha
    :param args: The data in *args has to follow this order: D, N, m_deg, nk. D is a matrix of 4 x m+1 x N dimensions,
                 N is the number of base sequences, m_deg is the degree of the highest moment, and nk is a 4 element
                 list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
                 projection)
    :return: a value
    '''
    D, N, m_deg, nk = args[0], args[1], args[2], args[3]
    n = sum(nk)
    sum_2 = 0
    sum_1 = 0
    for k in range(0, 4):
        sum_1 = 0
        for t in range(1, m_deg):
            sum_1 = n ** (m_deg - t) * comb(m_deg, t - 1) * (np.dot(x, D[k][0])) ** (m_deg - 1) * \
                    np.dot(x, D[k][m_deg - t + 1]) * (np.dot(x, D[k][1])) ** (t - 1)
        sum_2 += np.dot(x, D[k][0]) * (np.dot(x, D[k][1]) ** m_deg)
    return sum_1 + n * sum_2


def f_jac(x, *args):
    """
    Defines the gradient of the cost function.
    :param x: alpha
    :param args: The data in *args has to follow this order: D, N, m_deg, nk. D is a matrix of 4 x m+1 x N dimensions,
                 N is the number of base sequences, m_deg is the degree of the highest moment, and nk is a 4 element
                 list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
                 projection)
    :return: list with derivatives of each variable
    """
    D, N, m_deg, nk = args[0], args[1], args[2], args[3]
    n = sum(nk)
    der = [0] * N
    sum_1 = 0
    sum_2 = 0
    for i in range(0, N):
        for t in range(1, m_deg):
            sum_1 = 0
            t_coef = n ** (m_deg - t) * comb(m_deg, t - 1)
            for k in range(0, 4):
                sum_1 += nk[k] ** (m_deg - 1)*(D[k][m_deg - t + 1][i] * np.dot(x, D[k][1]) ** (t - 1) +
                         np.dot(x, D[k][m_deg - t + 1]) * (t - 1) * np.dot(x, D[k][1])  ** (t - 2) * D[k][1][i])
            sum_1 *= t_coef
            sum_2 = 0
            for k in range(0,4):
                sum_2 += nk[k] * m_deg * np.dot(x, D[k][1]) ** (m_deg-1) * D[k][1][i]
        der[i] = sum_1 + sum_2
    return der


def f_hess(x, *args):
    """
    Defines the hessian matrix for the cost function
    :param x: alpha
    :param args: The data in *args has to follow this order: D, N, m_deg, nk. D is a matrix of 4 x m+1 x N dimensions,
                 N is the number of base sequences, m_deg is the degree of the highest moment, and nk is a 4 element
                 list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
                 projection)
    :return: N x N matrix
    """
    D, N, m_deg, nk = args[0], args[1], args[2], args[3]
    hess = [[0 for i in range(N)] for j in range(N)]
    n = sum(nk)
    for i in range(N):
        for j in range(N):
            hij_1 = 0
            for t in range(1, m_deg):
                t_coef = n ** (m_deg - t) * comb(m_deg, t - 1)
                for k in range(4):
                    hij_1 += nk[k] ** (m - 1) * (D[k][m_deg - t + 1][i] * (t - 1) * np.dot(x, D[k][1]) ** (t - 2) * D[k][1][j] +
                            (t - 1) * D[k][1][i] * np.dot(x, D[k][m - t + 1]) * (t-2) * np.dot(x, D[k][1]) ** (t-3) * D[k][1][j] +
                                                 (t - 1) * D[k][1][i] * D[k][m - t + 1][j] * np.dot(x, D[k][1]) ** (t - 2))
                hij_1 *= t_coef
            hij_2 = 0
            for k in range(0, 4):
                hij_2 += nk[k] * D[k][1][i] * m_deg * (m_deg - 1) * np.dot(x, D[k][1]) ** (m_deg - 2) * D[k][1][j]
            hess[i][j] = hij_1 + hij_2
    return hess


def neg_f(x, *args):
    """
    Defines the negative cost function to be used in the maximizer
    :param x: alpha
    :param args: The data in *args has to follow this order: D, N, m_deg, nk. D is a matrix of 4 x m+1 x N dimensions,
                 N is the number of base sequences, m_deg is the degree of the highest moment, and nk is a 4 element
                 list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
                 projection)
    :return: a value
    """
    return -f(x, *args)


def neg_f_jac(x, *args):
    """
    Defines the negative jacobian function to be used in the maximizer
    :param x: alpha
    :param args: The data in *args has to follow this order: D, N, m_deg, nk. D is a matrix of 4 x m+1 x N dimensions,
                 N is the number of base sequences, m_deg is the degree of the highest moment, and nk is a 4 element
                 list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
                 projection)
    :return: the negative jacobian function
    """
    jac = f_jac(x, *args)
    return [-a for a in jac]


def neg_f_hess(x, *args):
    """
    Defines the negative hessian matrix to be used in the maximizer
    :param x: alpha
    :param args: The data in *args has to follow this order: D, N, m_deg, nk. D is a matrix of 4 x m+1 x N dimensions,
                 N is the number of base sequences, m_deg is the degree of the highest moment, and nk is a 4 element
                 list, with each element representing nA, nC, nG, and nT, respectively (determined by the 4-d
                 projection)
    :return: the negative hessian matrix
    """
    hess = np.array(f_hess(x, *args))
    return -1 * hess


if __name__ == "__main__":
    # enter in terminal "python solve.py m", where m >= 2
    m = int(sys.argv[1])
    print(m)
    #the number of As, Cs, Ts, and Gs. To be tested
    test_nk = [2890, 1436, 1817, 1875]
    nk = test_nk
    n = sum(test_nk)
    # generate natural vector file
    save_natural_vector('group_M_shortest20.fasta', 'vectors_1.txt', m)
    sequence_file = 'C:\\Users\\pizza\\PycharmProjects\\Bioinformatics\\vectors_1.txt'
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
    additional = (D, N, m, test_nk)
    m_deg = m
    lin_con_matrix = linear_constraint_matrix()
    linear_constraint = LinearConstraint(lin_con_matrix, linear_bound(test_nk), linear_bound(test_nk))
    bounds = Bounds([0] * N, [1] * N)
    #initial guess of alpha
    x0 = np.asarray([1 / N] * N)

    if m == 2:
        opt_constraints = [linear_constraint]
    elif m >= 3:
        nonlinear_constraint = NonlinearConstraint(non_linear_const_matrix, nonlinear_bound(test_nk, m),
                                                   nonlinear_bound(test_nk, m), jac=nonlinear_jac_matrix,
                                                   hess=nonlinear_hess_matrix)
        opt_constraints = [linear_constraint, nonlinear_constraint]
    else:
        print("Invalid m value.")
        quit()

    min_res = minimize(f, x0, args=additional, method='trust-constr', jac=f_jac, hess=f_hess,
                       constraints=opt_constraints, options={'verbose':1}, bounds=bounds)
    print(min_res.x)

    max_res = minimize(neg_f, x0, args=additional, method='trust-constr', jac=neg_f_jac, hess=neg_f_hess,
                      constraints=opt_constraints, options={'verbose': 1}, bounds=bounds)
    print(max_res.x)
    sm = exp_sum(n, m)
    print(min_res.fun <= sm <= -max_res.fun)
    print('sum=%s' % sm)
    print('min_f=%s' % min_res.fun)
    print('max_f=%s' % -max_res.fun)
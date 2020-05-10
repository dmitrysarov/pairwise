import numpy as np
from itertools import combinations
from scipy import sparse
import matplotlib.pyplot as plt


class Pair_comp(object):
    '''
    perform comparison procedure and return absolute ranking of input samples
    '''

    @staticmethod
    def make_index_matrix(DIM):
        '''
        this is helper function for likelihood calculation
        matrix of pairwise comparison indeces
        create matrix where each element is list of binary index,
        where 1 placed at position [i] and [j] of array of length DIM,
        where i,j is matrix row and colomn
        don't care about diag elem (futher they not used)
        Example: matrix for DIM = 3 will look like
        [[[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]],
         [[1., 1., 0.], [0., 1., 0.], [0., 1., 1.]],
         [[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]])
        '''
        matrix = []
        for i in range(DIM):
            row = []
            for j in range(DIM):
                elem = [0]*DIM
                elem[i] = 1
                elem[j] = 1
                row.append(elem)
            matrix.append(row)
        return np.array(matrix, dtype=np.float32)


    def __init__(self, obj_arr, comparison_densety='dense'):
        '''
        param: obj_arr (size: num_elem x ...; type: numpy array) array of objects to be compared
        param: comparison_densety {'dense', 'sparse'}. Whether to compare obj with each other or
        in other faster way where indeces schocasticaly choosen.
        '''
        self.obj_arr = obj_arr
        self.comparison_densety = comparison_densety
        self.comp_matrix = np.zeros((len(obj_arr), len(obj_arr)))
        self.obj_dim = len(obj_arr[0].shape) # 1D - for test purpose, 2D - image, 3D - video (TODO)
        self.ind_matrix = self.make_index_matrix(len(self.obj_arr))
        self.scores = np.ones(len(obj_arr)) # default scores

    def sample_pair_sparse(self, additive=1):
        '''
        stochastic solution of sample task
        heuristic is that number of samples should be larger then object numbers by 1 or more ["additive" param]
        '''
        raise NotImplementedError('Not yet implemented, use comparison_densety="dense"')

    def sample_pair_dense(self):
        '''
        sampled all combinations of objects without replacemnet
        '''
        for ind1, ind2 in combinations(range(len(self.obj_arr)), r=2): #e.g. 6 samples - 15 comparisons
            yield ind1, ind2

    def compare_pair(self, ind1, ind2):
        '''
        perform comparison of objects. Input value should be {-1, 0, 1}
        '''
        if self.obj_dim == 1:
            print('object 1: {}; object 2: {}'.format(self.obj_arr[ind1], self.obj_arr[ind2]))
        elif self.obj_dim == 2:
            _, ax = plt.subplots(1, 2, figsize=(5, 5))
            ax[0].imshow(self.obj_arr[ind1])
            ax[0].set_title('object 1')
            ax[1].imshow(self.obj_arr[ind2])
            ax[1].set_title('object 2')
            plt.show()
        else:
            raise NotImplementedError('object dimentiality not supported')
        comp_result = input('Compare objects (-1: object 1 win; 0: draw; 1: object 2 win)')
        return comp_result # {-1, 0, 1}

    def show_objects(self):
        '''
        print of depict objects
        '''
        if self.obj_dim == 1:
            for i in range(len(self.obj_arr)):
                print('Object {}: {}'.format(i, self.obj_arr[i]), '\n Score {:.4f}'.format(self.scores[i]))
        elif self.obj_dim == 2:
            _, ax = plt.subplots(1, len(self.obj_arr))
            for i in range(len(self.obj_arr)):
                ax[i].imshow(self.obj_arr[i])
                ax[i].set_title('Object {} \n Score {:.4f}'.format(i, self.scores[i]))
            plt.tight_layout()
            plt.show()
        else:
            raise NotImplementedError('object dimentiality not supported')

    def compare_samples(self):
        '''
        execute comparison procedure
        '''
        if self.comparison_densety == 'dense':
            sampler = self.sample_pair_dense()
        elif self.comparison_densety == 'sparse':
            sampler = self.sample_pair_sparse()
        else:
            raise
        for ind1, ind2 in sampler:
            winner = int(self.compare_pair(ind1, ind2)) # on of {-1, 0, 1}; -1 ind1 win, 0 equal, 1 ind2 win
            if winner == -1:
                self.comp_matrix[ind1, ind2] += 1
            elif winner == 0:
                self.comp_matrix[ind1, ind2] += 1
                self.comp_matrix[ind2, ind1] += 1
            elif winner == 1:
                self.comp_matrix[ind2, ind1] += 1
            else:
                raise

    def calc_loglikelihood(self, scores, comp_matrix):
        '''
        competition_matrix - where counted wins of i-th row over j-th colomn
        scores - initial scores, used as optimization parameter
        '''
        scores_matrix = scores.reshape(-1,1)/(self.ind_matrix*scores + 1e-10).sum(-1) #
        return (comp_matrix*np.log(scores_matrix + 1e-10)).sum()

    def fit_bt(self):
        '''
        fit Bradley-Terry model
        return scores values
        '''
        self.ll_log = []
        scores = np.ones(len(self.obj_arr))
        def update_score(competition_matrix, scores):
            assert all(np.diag(competition_matrix) == 0)
            m = (competition_matrix + competition_matrix.transpose(0,1))/(self.ind_matrix*scores + 1e-10).sum(-1)
            m = m - np.diag(np.diag(m))
            scores = competition_matrix.sum(-1)/(m + 1e-10).sum(-1)
            return scores/scores.sum()
        prev_ll = -np.inf
        ll = self.calc_loglikelihood(scores, self.comp_matrix)
        self.ll_log.append(ll)
        while True:
            scores = update_score(self.comp_matrix, scores)
            ll = self.calc_loglikelihood(scores, self.comp_matrix)
            if np.isclose(prev_ll, ll, rtol=1e-6) or prev_ll>ll: #plateau
                break
            prev_ll = ll
            self.ll_log.append(ll)
        self.scores = scores

    def add_object(self, obj):
        '''
        add new object and compare it with best scored till this moment
        '''
        self.obj_arr = np.concatenate((self.obj_arr, obj[np.newaxis, ...]), axis=0)
        self.ind_matrix = self.make_index_matrix(len(self.obj_arr))
        comp_matrix = np.zeros((len(self.obj_arr), len(self.obj_arr)))
        comp_matrix[:-1,:-1] = self.comp_matrix # add new row and column
        self.comp_matrix = comp_matrix
        ind1 = len(self.obj_arr) - 1 # last object
        ind2 = np.argmax(self.scores) # best object
        winner = int(self.compare_pair(ind1, ind2)) # on of {-1, 0, 1}; -1 ind1 win, 0 equal, 1 ind2 win
        if winner == -1:
            self.comp_matrix[ind1, ind2] += 1
        elif winner == 0:
            self.comp_matrix[ind1, ind2] += 1
            self.comp_matrix[ind2, ind1] += 1
        elif winner == 1:
            self.comp_matrix[ind2, ind1] += 1
        else:
            raise


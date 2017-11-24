from nltk.translate.gale_church import align_log_prob

cdef extern from "sgd_mf.h":
    cdef struct LogData:
        int item_id
        int user_id
        double value
        
    cdef cppclass SgdMf:
        SgdMf(int n_iter, double lamb, int num_feature)
        
cdef class SgdMatroxFactorization:
    cdef SgdMf *matrix_factoriation
    
    def __init__(self, int n_iter, double lamb, int num_feature):
        self.matrix_factoriation = new  SgdMf(n_iter, lamb, num_feature)
    
    def fit(self, corpus):
        cdef int n_logs, n_samples, n_dimentions
        n_samples = len(corpus)
        n_logs = n_samples = n_dimentions = 0
        for row_corpus in corpus:
            for w_i, w_c in row_corpus:
                n_logs += 1
                if n_dimentions > w_i:
                    n_dimentions = w_i
        n_dimentions += 1
        
        self.matrix_factoriation.fit(sample_X, n_logs, n_samples, n_dimentions)
        
    
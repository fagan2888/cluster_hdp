from sklearn.utils import check_random_state
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class HdpTopic:
    """HDP topic model as defined in Teh et al.

    Parameters
    ----------
    alpha0: float
        Concentration parameter
    gamma: float
        Concentration parameter
    iter: int
        Number of iterations for fitting
    random_state: RandomState or int or None (default: None)
        Random state
    """
    def __init__(self, alpha0, gamma, iter, random_state=None):
        self.alpha0 = alpha0
        self.omega0 = []# Code des densites initiales
        self.gamma = gamma
        self.vocabulary_size = -1
        self.iter = iter
        self.random_state = check_random_state(random_state)
        self.n_obs_kw_ = {}
        self.n_obs_jk_ = {}
        self.beta_ = {}  # NB: self._beta[-1] is beta_u
        self.m_j_k_ = []
        self.z_ji_ = []
        self.n_topics_ = 0
        self._precomputed_stirling = {}
        self._data = []

    def _set_vocabulary_size(self):
        """Set vocabulary size according to the data provided at fit time.

        Examples
        --------
        >>> docs = [[1, 0, 2]]
        >>> model = HdpTopic(alpha0=1., gamma=1., iter=0)
        >>> model.fit(docs)
        >>> model.vocabulary_size
        3
        >>> docs = [[1, 3, 5], [2, 4], [1, 8]]
        >>> model = HdpTopic(alpha0=1., gamma=1., iter=0)
        >>> model.fit(docs)
        >>> model.vocabulary_size
        9
        """
        for j in range(self.n_docs_):
            for i in range(self.n_obs_in_doc(j)):
                if self._data[j][i] >= self.vocabulary_size:
                    self.vocabulary_size = self._data[j][i] + 1

    @property# Attribut
    def n_docs_(self):
        return len(self._data)

    def n_obs_in_doc(self, j):
        """Number of observations in a given document.

        Parameter
        ---------
        j: int
            Document index

        Returns
        -------
        int
            Number of observations in document j ($N_j$)

        Example
        -------
        >>> docs = [[1, 3, 5], [2, 4], [1, 8]]
        >>> model = HdpTopic(alpha0=1., gamma=1., iter=0)
        >>> model.fit(docs)
        >>> model.n_obs_in_doc(0)
        3
        >>> model.n_obs_in_doc(1)
        2
        >>> model.n_obs_in_doc(2)
        2
        """
        return len(self._data[j])

    def n_k_w(self, k, w):
        """Number of observations for a given word associated to a given topic.

        Parameter
        ---------
        k: int
            Topic index
        w: int
            Word index

        Returns
        -------
        int
            Number of observations
        """
        if k < 0:
            return 0
        return self.n_obs_kw_[k, w]

    def n_j_k(self, j, k):
        """Number of observations in a given document associated to a given topic.

        Parameter
        ---------
        j: int
            Document index
        k: int
            Topic index

        Returns
        -------
        int
            Number of observations
        """
        if k < 0:
            return 0
        return self.n_obs_jk_[j, k]

    def f_k_w(self, k, w):
        """Compute function $f_k$ as defined in Eq.(30) in Teh et al.

        Parameter
        ---------
        k: int
            Topic index
        w: int
            Word index

        Returns
        -------
        float
            $f_k(w)$
        """
        return self.omega0[w] + self.n_k_w(k, w)

    def m_k(self, k):
        """Number of occurrences for a given topic ($m_{\cdot, k}$ in Teh et al.)

        Parameter
        ---------
        k: int
            Topic index

        Returns
        -------
        int
            Number of occurrences
        """
        if k < 0:
            return 0
        return sum([m_j[k] for m_j in self.m_j_k_])

    def draw_beta(self):
        """Draw a vector of betas from Eq.(36) in Teh et al.)

        Returns
        -------
        ndarray
            Vector of betas of size n_topics + 1, in which beta_u is last
        """
        alpha = [self.m_k(k) for k in range(self.n_topics_)] + [self.gamma]
        return self.random_state.dirichlet(alpha=alpha)

    def _stirling(self, n, m):
        """Compute Stirling function term based on cached values.

        Paramewters
        -----------
        n: int
        m: int

        Returns
        -------
        int
            s(n, m)
        """
        if (n, m) in self._precomputed_stirling.keys():
            return self._precomputed_stirling[n, m]
        elif m > n:
            return 0
        elif m == n and (m == 0 or m == 1):
            return 1
        elif m == 0:
            return 0
        else:
            self._precomputed_stirling[n, m] = self._stirling(n - 1, m - 1) + (n - 1) * self._stirling(n - 1, m)
            return self._precomputed_stirling[n, m]

    def draw_m_jk(self, j, k):
        """Draw number of occurrences $m_{j,k}$ according to Eq.(40) in Teh et al.)

        Parameter
        ---------
        j: int
            Document index
        k: int
            Topic index

        Returns
        -------
        int
            Drawn number of occurrences
        """
        probas = []
        n_j_k = self.n_j_k(j, k)
        for m in range(n_j_k + 1):
            probas.append(self._stirling(n_j_k, m) * (self.alpha0 * self.beta_[k]) ** m)
        probas = numpy.array(probas, dtype=numpy.float64)
        probas /= (1e-9 + numpy.sum(probas))
        draw = self.random_state.multinomial(n=1, pvals=probas)
        return numpy.argmax(draw)

    def draw_z_ji(self, j, i):
        """Draw z_ji based on Eq.(37) from Teh et al.

        Parameters
        ----------
        j: int
            Document index
        i: int
            Observation index (inside document `j`)

        Returns
        -------
        int
            Drawn topic assignment z_ji for observation x_ji
        """
        w = self._data[j][i]
        probas = []
        for k in range(self.n_topics_):
            pr = (self.n_j_k(j, k) + self.alpha0 * self.beta_[k]) * self.f_k_w(k, w)
            probas.append(pr)
        probas.append(self.alpha0 * self.beta_[-1] * self.f_k_w(-1, w))
        probas = numpy.array(probas, dtype=numpy.float64)
        probas /= (1e-9 + numpy.sum(probas))
        draw = self.random_state.multinomial(n=1, pvals=probas)
        return numpy.argmax(draw)

    def update_counts(self, j, i, increment):
        """Update counts for a given observation (withdraw it or add it in the
        counts).

        This function is called before drawing a z_ji (so that we perform -(ji)
        on all counts) and after a new z_ji value is drawn (to take it into
        account for other z values to be drawn).
        Does not do anything if z_ji is `None` (initial state).

        Parameters
        ----------
        j: int
            Document index
        i: int
            Observation index (inside document `j`)
        increment: int
            Amount to be added to the related counts
        """
        if self.z_ji_[j][i] is not None:
            k = self.z_ji_[j][i]
            self.n_obs_jk_[j, k] = self.n_obs_jk_.get((j, k), 0) + increment
            w = self._data[j][i]
            self.n_obs_kw_[k, w] = self.n_obs_kw_.get((k, w), 0) + increment

    def fit_one_iter(self):
        """Do a single iteration of the Gibbs sampling process."""
        # Draw z_ji
        for j in range(self.n_docs_):
            for i in range(self.n_obs_in_doc(j)):
                # a. Unset z_ji
                self.update_counts(j, i, -1)
                # b. Draw z_ji
                self.z_ji_[j][i] = self.draw_z_ji(j, i)
                if self.z_ji_[j][i] >= self.n_topics_:
                    self.n_topics_ += 1
                # c. Update counts
                self.update_counts(j, i, 1)

        # Draw m_jk
        self.m_j_k_ = []
        for j in range(self.n_docs_):
            self.m_j_k_.append([])
            for k in range(self.n_topics_):
                self.m_j_k_.append(self.draw_m_jk(j, k))

        # Draw beta
        betas = self.draw_beta()
        self.beta_[-1] = betas[-1]
        for k in range(self.n_topics_):
            self.beta_[k] = betas[k]

    def fit(self, X):
        """Fit the model to the data provided in `X`.

        Parameters
        ----------
        X: list of lists of integers
            Observations (list of documents, each document being itself a list
            of obersvations). Each observation is an integer between 0 and |V|-1
        """
        self._data = X
        self._set_vocabulary_size()
        for word in range(self.vocabulary_size):
            self.omega0[word] = 1
        self.z_ji_ = []
        for j in range(self.n_docs_):
            self.z_ji_.append([None] * self.n_obs_in_doc(j))
        betas = self.draw_beta()
        self.beta_[-1] = betas[-1]

        for it in range(self.iter):
            self.fit_one_iter()


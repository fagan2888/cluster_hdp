import numpy

class MixtureModel:
    """
    Parameters
    ----------
    movement_modes: numpy array
        weights over the words.
        The list is made of n_movement_modes elements
        Each element should have same size, say dict_size and should be positive elements summing to one
    """
    def __init__(self, movement_modes):
        self.movement_modes = movement_modes
        self.n_movement_modes = movement_modes.shape[0]
        self.dict_size = movement_modes.shape[1]

    def generate_documents(self, n_docs, n_words):
        """
        Parameters
        ----------
        n_docs: int
            number of documents
        n_words, int
            number of words per document
        """
        parameter_dirichlet = numpy.ones(self.n_movement_modes)
        documents = numpy.full((n_docs, n_words), -1)

        for doc in range(n_docs):
            movement_modes_weights = numpy.random.dirichlet(parameter_dirichlet, 1)[0]
            for word in range(n_words):
                mode_number = numpy.random.choice(range(self.n_movement_modes), size=1, p=movement_modes_weights)[0]
                documents[doc, word] = numpy.random.choice(range(self.dict_size), size=1,
                                                           p=self.movement_modes[mode_number])[0]
        return documents.tolist();
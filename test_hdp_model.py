import numpy
import hdp_model
import data_gen

## TEST Phase

n_words = 6
n_movement_modes = 3
numpy.random.seed(2)
alphas = numpy.full((n_words), 1)
movement_modes = numpy.random.dirichlet(alphas, n_movement_modes)
modele = data_gen.MixtureModel(movement_modes)
toy_data = modele.generate_documents(n_docs=20, n_words=100)

modele = hdp_model.HdpTopic(1, 1, 20)
modele.fit(toy_data)
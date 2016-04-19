# Run sudo pip install annoy

from annoy import AnnoyIndex
import random

class ANN:
    def __init__(self, dimension):
        self.ann = AnnoyIndex(dimension)
    def addVectors(self,vectors):
        for idx,v in enumerate(vectors):
            self.ann.add_item(idx,v)
        self.ann.build(10)
    def query(self,vector):
        match = self.ann.get_nns_by_vector(vector,1)[0]
        # return self.ann.get_item_vector(match),match
        return match

# if __name__ == "__main__":
#     f = 40
#     ann = ANN(f)
#     ann.addVectors([[random.gauss(0, 1) for z in xrange(f)] for x in xrange(1000)])
#     print(ann.query([random.gauss(0, 1) for z in xrange(f)]))

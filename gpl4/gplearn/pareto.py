from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedKeyList
from sklearn.cluster import KMeans as KNN


class Point(object):
    def __init__(self, x, y, data=None, id=None):
        self.x = x
        self.y = y
        self.data = data
        self.id = id
        self.cluster = None

    def __getitem__(self, index):
        """Indexing: get item according to index."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.data
        elif index == 3:
            return self.id
        else:
            raise Exception("Index {} is out of range!".format(index))


    def __setitem__(self, index, value):
        """Indexing: set item according to index."""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.data = value
        elif index == 3:
            raise Exception("Cannot set Id!")
        else:
            raise Exception("Index {} is out of range!".format(index))

    
class ParetoSet(SortedKeyList):
    """Maintained maximal set with efficient insertion. Note that we use the convention of smaller the better."""

    def __init__(self,num_clusters=5):
        super().__init__(key=lambda p: p.x)
        self.num_clusters = num_clusters
        self.clustering = []
        self.cluster_model = KNN(n_clusters = self.num_clusters, random_state=0, n_init="auto")
        self.fitted = False
        
    def _input_check(self, p):
        """Check that input is in the correct format.

        Args:
            p: input

        Returns:
            Point:

        Raises:
            TypeError if cannot be converted.
        """

        if isinstance(p, Point):
            return p
        elif isinstance(p, tuple) and len(p) == 2:
            return Point(x=p[0], y=p[1], data=None)
        else:
            raise TypeError("Must be instance of Point or 2-tuple.")
    
    
    def get_id_list(self):
        id_list = []
        for point in self:
            id_list.append(point.id)
        return id_list


    def recluster(self):
        '''
        x_grad = (self[-1]-self[0])/self.num_clusters
        clust_num = 0
        lower = x_grad
        for p in self:
            if p.x < lower:
                p.cluster = clust_num
            else:
                clust_num +=1
                lower += x_grad
                p.cluster = clust_num
        '''
        self.cluster_model.fit(self.to_array())
        self.clustering = list(self.cluster_model.labels_)

    def add(self, p):
        """Insert Point into set if minimal in first two indices.

        Args:
            p (Point): Point to insert

        Returns:
            bool: True only if point is inserted

        """
        p = self._input_check(p)

        is_pareto = False
        # check right for dominated points:
        right = self.bisect_left(p)

        while len(self) > right and self[right].y >= p.y and not (self[right].x == p.x and self[right].y == p.y):
            self.pop(right)
            is_pareto = True

        # check left for dominating points:
        left = self.bisect_right(p) - 1

        if left == -1 or self[left][1] > p[1]:
            is_pareto = True

        # if it's the only point it's maximal
        if len(self) == 0:
            is_pareto = True

        if is_pareto:
            super().add(p)
            if len(self) > 5:
                if p.x < self[0].x or p.x > self[-1].x or not self.fitted:
                    self.recluster()
                else:
                    '''
                    x_grad = (self[-1].x - self[0].x)/self.num_clusters
                    lower = self[0].x
                    for i in range(self.num_clusters):
                        if p.x < lower:
                            p.cluster = i
                            break
                        lower+=x_grad
                    '''
                    c = self.cluster_model.predict([p.x,p.y])
                    self.clustering.insert(right,c)

        return is_pareto


    def __contains__(self, p):
        p = self._input_check(p)

        left = self.bisect_left(p)

        while len(self) > left and self[left].x == p.x:
            if self[left].y == p.y:
                return True

            left += 1

        return False


    def __add__(self, other):
        """Merge another pareto set into self.

        Args:
            other (ParetoSet): set to merge into self

        Returns:
            ParetoSet: self

        """

        for item in other:
            self.add(item)

        return self


    def distance(self, p):
        """Given a Point, calculate the minimum Euclidean distance to pareto
        frontier (in first two indices).

        Args:
            p (Point): point

        Returns:
            float: minimum Euclidean distance to pareto frontier

        """
        p = self._input_check(p)

        point = np.array((p.x, p.y))
        dom = self.dominant_array(p)

        # distance is zero if pareto optimal
        if dom.shape[0] == 0:
            return 0.

        # add corners of all adjacent pairs
        candidates = np.zeros((dom.shape[0] + 1, 2))
        for i in range(dom.shape[0] - 1):
            candidates[i, :] = np.max(dom[[i, i+1], :], axis=0)

        # add top and right bounds
        candidates[-1, :] = (p.x, np.min(dom[:, 1]))
        candidates[-2, :] = (np.min(dom[:, 0]), p.y)

        return np.min(np.sqrt(np.sum(np.square(candidates - point), axis=1)))


    def dominant_array(self, p):
        """Given a Point, return the set of dominating points in the set (in
        the first two indices).

        Args:
            p (Point): point

        Returns:
            numpy.ndarray: array of dominating points

        """
        p = self._input_check(p)

        idx = self.bisect_left(p) - 1

        domlist = []

        while idx >= 0 and self[idx][1] < p[1]:
            domlist.append(self[idx])
            idx -= 1

        return np.array([x[0:2] for x in domlist])


    def to_array(self):
        """Convert first two indices to numpy.ndarray

        Args:
            None

        Returns:
            numpy.ndarray: array of shape (len(self), 2)

        """
        A = np.zeros((len(self), 2))
        for i, p in enumerate(self):
            A[i, :] = p.x, p.y

        return A

    def get_pareto_points(self):
        """Returns the x, y and data for each point in the pareto frontier
        
        """
        pareto_points = []
        for i, p in enumerate(self):
            pareto_points = pareto_points + [[p.x, p.y, p.data]]
        
        return pareto_points
        

    def from_list(self, A):
        """Convert iterable of Points into ParetoSet.

        Args:
            A (iterator): iterator of Points

        Returns:
            None

        """
        for a in A:
            self.add(a)
    
    
    def plot(self):
        """Plotting the Pareto frontier."""
        array = self.to_array()
        plt.figure(figsize=(8, 6))
        plt.plot(array[:, 0], array[:, 1], 'r.')
        plt.show()



    def prune(self, min_points=15, max_points = 25):
        """ remove less useful solutions from the Pareto front
remove points proportionally from each cluster
        """
        num_points_remove = int((len(self)-min_points)/self.num_clusters)
        if len(self) > max_points and len(self) > 5:
            
            cluster_counts = np.array([[x,self.clustering.count(x)] for x in set(self.clustering)])
            cluster_counts[:,1] = (cluster_counts[:,1]/(self.num_clusters)*num_points_remove).astype(int)
            
            for i in range(cluster_counts.shape[0]):
                clustering_arr = np.array(self.clustering)
                ii = np.where(clustering_arr == i)[0]
                items_delete = np.random.choice(ii,cluster_counts[i,1],replace=False)
                items_delete[::-1].sort()
                for item in items_delete:
                    self.clustering.pop(item)
                    self.pop(item)
                    
            
            
        
if __name__ == "__main__":
    PA = ParetoSet()
    A = np.zeros((40, 2))
    
    for i in range(40):
        x = np.random.rand()
        y = np.random.rand()
        
        A[i, 0] = x
        A[i, 1] = y
        
        PA.add(Point(x=x, y=y, data=None))
    print(A)
    paretoA = PA.to_array()
    print(paretoA)

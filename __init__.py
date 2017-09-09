import unittest

import torch
from torch.autograd import Variable, Function
import knn_pytorch


class KNearestNeighbor(Function):
  """Accumulate x += y using broadcasting sum.
  """
  def __init__(self, k):
    self.k = k

  def forward(self, ref, query):
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = torch.zeros(self.k, query.shape[1]).long().cuda()
    dists = torch.zeros(self.k, query.shape[1]).float().cuda()

    knn_pytorch.knn(ref, query, inds, dists)

    return inds, dists


class TestKNearestNeighbor(unittest.TestCase):

  def test_forward(self):
    D, N, M = 128, 100, 1000
    ref = Variable(torch.rand(D, N))
    query = Variable(torch.rand(D, M))

    inds, dists = KNearestNeighbor(2)(ref, query)
    print inds, dists


if __name__ == '__main__':
  unittest.main()

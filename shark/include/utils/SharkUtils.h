#pragma once

#include "MooseUtils.h"
#include "RankTwoTensor.h"
#include "RankFourTensor.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/dense_matrix.h"
#include <unordered_map>

#ifdef LIBTORCH_ENABLED
#include <torch/torch.h>
#endif

namespace SharkUtils
{
void fillSymmetricRankFourFromInputVector(RankFourTensor & x, const std::vector<Real> & input);

#ifdef LIBTORCH_ENABLED

void fillSymmetricRankFourFromInputTensor(RankFourTensor & x,
                                          const torch::Tensor & input,
                                          const bool isEngineeringStrain = true);

inline torch::Tensor
tensorFromSymmetricRankFour(const RankFourTensor & x, c10::ScalarType dtype)
{
  return torch::tensor(
      {{x(0, 0, 0, 0), x(0, 0, 1, 1), x(0, 0, 2, 2), x(0, 0, 1, 2), x(0, 0, 0, 2), x(0, 0, 0, 1),
        x(1, 1, 0, 0), x(1, 1, 1, 1), x(1, 1, 2, 2), x(1, 1, 1, 2), x(1, 1, 0, 2), x(1, 1, 0, 1),
        x(2, 2, 0, 0), x(2, 2, 1, 1), x(2, 2, 2, 2), x(2, 2, 1, 2), x(2, 2, 0, 2), x(2, 2, 0, 1),
        x(1, 2, 0, 0), x(1, 2, 1, 1), x(1, 2, 2, 2), x(1, 2, 1, 2), x(1, 2, 0, 2), x(1, 2, 0, 1),
        x(0, 2, 0, 0), x(0, 2, 1, 1), x(0, 2, 2, 2), x(0, 2, 1, 2), x(0, 2, 0, 2), x(0, 2, 0, 1),
        x(0, 1, 0, 0), x(0, 1, 1, 1), x(0, 1, 2, 2), x(0, 1, 1, 2), x(0, 1, 0, 2), x(0, 1, 0, 1)}},
      dtype);
}

void fillSymmeticRankTwoFromInputTensor(RankTwoTensor & x, const torch::Tensor & input);

void fillRankTwoFromInputTensor(RankTwoTensor & x, const torch::Tensor & input);

void fillRankFourFromInputTensor(RankFourTensor & x, const torch::Tensor & input);

template <typename T>
inline void
fillVectorFromTensor(std::vector<T> & x, const torch::Tensor & input)
{
  auto size = input.numel();
  x.resize(size);
  std::copy(input.data_ptr<T>(), input.data_ptr<T>() + size, x.begin());
}

RankFourTensor SymmetricRankFourTensorFromTensor(const torch::Tensor & input, unsigned int dim = 3);
#endif

template <typename T>
inline DenseMatrix<T>
denseMatrix(const SparseMatrix<T> & mat)
{
  DenseMatrix<T> dm(mat.m(), mat.n());
  for (dof_id_type i = 0; i < mat.m(); i++)
    for (dof_id_type j = 0; j < mat.n(); j++)
      dm(i, j) = mat(i, j);

  return dm;
}

// use lu_solve() to calculate A^{-T}
// A A^{-1} = I
// return A^{-T}
template <typename T>
inline DenseMatrix<T>
denseMatrixRightInverseTransposeLU(const DenseMatrix<T> & mat, unsigned int num_threads = 1)
{
  DenseMatrix<T> m(mat);
  auto n = m.n();
  mooseAssert(m.m() == n, "Not a square matrix, cannot find the inverse!");

  DenseMatrix<T> inverse_mat_T(n, n);

  auto task = [&](unsigned int start, unsigned int end)
  {
    for (unsigned int i = start; i < end; i++)
    {
      DenseVector<T> x(n);
      DenseVector<T> e(n, 0.0);
      e(i) = 1.0;
      m.lu_solve(e, x);

      std::move(
          x.get_values().begin(), x.get_values().end(), inverse_mat_T.get_values().begin() + i * n);
    }
  };

  // Compute the last row (row n-1) and perform LU decomposition.
  task(n - 1, n);

  if (num_threads == 1)
  {
    task(0, n - 1);
    return inverse_mat_T;
  }

  // Limit the number of threads to avoid excessive threading overhead on small matrices
  num_threads = std::min(num_threads, n - 1);

  std::vector<std::thread> threads;
  // (n - 1) - 1   e.g.  n=4, num_threads=2, chunk_size=2; n=5, num_threads=2, chunk_size=3; n=6,
  // num_threads=2, chunk_size=3
  unsigned int chunk_size = (n - 2) / num_threads + 1;
  for (unsigned int i = 0; i < num_threads; ++i)
  {
    unsigned int start = i * chunk_size;
    unsigned int end = std::min(start + chunk_size, n - 1);
    threads.emplace_back(task, start, end);
  }

  for (auto & t : threads)
  {
    if (t.joinable())
    {
      t.join();
    }
  }
  return inverse_mat_T;
}

// use lu_solve() to calculate A^{-1} : A^{-1} A = I
template <typename T>
inline DenseMatrix<T>
denseMatrixLeftInverseLU(const DenseMatrix<T> & mat, unsigned int num_threads = 1)
{
  DenseMatrix<T> inverse_mat(mat.n(), mat.m());
  mat.get_transpose(inverse_mat);
  return denseMatrixRightInverseTransposeLU(inverse_mat, num_threads);
}

template <typename T>
inline DenseVector<T>
denseMatrixRow(const DenseMatrix<T> & mat, unsigned int row)
{
  mooseAssert(row <= mat.m(), "row exceeds the maximum number of rows!");
  auto n = mat.n();
  DenseVector<T> v(n);
  std::copy(mat.get_values().begin() + row * n,
            mat.get_values().begin() + (row + 1) * n,
            v.get_values().begin());
  return v;
}

template <typename T>
inline DenseVector<T>
denseMatrixColumn(const DenseMatrix<T> & mat, unsigned int column)
{
  mooseAssert(column <= mat.n(), "column exceeds the maximum number of columns!");
  auto m = mat.m();
  DenseVector<T> v(m);
  for (unsigned int i = 0; i < m; i++)
    v(i) = mat(i, column);

  return v;
}

class dofMap
{
public:
  void insert(unsigned int node_id, std::vector<unsigned int> dof_ids);
  std::vector<unsigned int> getDofIds(unsigned int node_id) const;
  unsigned int getNodeId(unsigned int dof_id) const;
  void clear();
  void print(std::ostream & os = std::cout);

private:
  std::unordered_map<unsigned int, std::vector<unsigned int>> _dof_ids;
  std::unordered_map<unsigned int, unsigned int> _node_id;
};

class nodeMap
{
public:
  void insert(unsigned int node_id, unsigned int map_id);
  unsigned int getMapId(unsigned int node_id) const;
  unsigned int getNodeId(unsigned int map_id) const;
  bool hasNodeId(unsigned int node_id) const;
  void clear();
  void print(std::ostream & os = std::cout);

private:
  std::unordered_map<unsigned int, unsigned int> _map_id;
  std::unordered_map<unsigned int, unsigned int> _node_id;
};

RankFourTensor SymmetricRankFourTensorFromInputVector(const std::vector<Real> & input,
                                                      unsigned int dim = 3);

std::vector<Real> VectorFromInputSymmetricTensor(const RankFourTensor & input,
                                                 unsigned int dim = 3);

std::vector<Real> VectorFromInputSymmetricTensor(const RankTwoTensor & input, unsigned int dim = 3);

}
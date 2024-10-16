#include "SharkUtils.h"
#include "DualRealOps.h"
#include "RankTwoTensorImplementation.h"
namespace SharkUtils
{
void
fillSymmetricRankFourFromInputVector(RankFourTensor & x, const std::vector<Real> & input)
{
  // if (input.size() != 36)
  //   mooseError("Input vector must be of size 36 for symmetric rank 4 tensor.\n");

  x(0, 0, 0, 0) = input[0];
  x(0, 0, 1, 1) = input[1];
  x(0, 0, 2, 2) = input[2];
  x(0, 0, 1, 2) = x(0, 0, 2, 1) = input[3];
  x(0, 0, 0, 2) = x(0, 0, 2, 0) = input[4];
  x(0, 0, 0, 1) = x(0, 0, 1, 0) = input[5];

  x(1, 1, 0, 0) = input[6];
  x(1, 1, 1, 1) = input[7];
  x(1, 1, 2, 2) = input[8];
  x(1, 1, 1, 2) = x(1, 1, 2, 1) = input[9];
  x(1, 1, 0, 2) = x(1, 1, 2, 0) = input[10];
  x(1, 1, 0, 1) = x(1, 1, 1, 0) = input[11];

  x(2, 2, 0, 0) = input[12];
  x(2, 2, 1, 1) = input[13];
  x(2, 2, 2, 2) = input[14];
  x(2, 2, 1, 2) = x(2, 2, 2, 1) = input[15];
  x(2, 2, 0, 2) = x(2, 2, 2, 0) = input[16];
  x(2, 2, 0, 1) = x(2, 2, 1, 0) = input[17];

  x(1, 2, 0, 0) = x(2, 1, 0, 0) = input[18];
  x(1, 2, 1, 1) = x(2, 1, 1, 1) = input[19];
  x(1, 2, 2, 2) = x(2, 1, 2, 2) = input[20];
  x(1, 2, 1, 2) = x(1, 2, 2, 1) = x(2, 1, 1, 2) = x(2, 1, 2, 1) = input[21];
  x(1, 2, 0, 2) = x(1, 2, 2, 0) = x(2, 1, 0, 2) = x(2, 1, 2, 0) = input[22];
  x(1, 2, 0, 1) = x(1, 2, 1, 0) = x(2, 1, 0, 1) = x(2, 1, 1, 0) = input[23];

  x(0, 2, 0, 0) = x(2, 0, 0, 0) = input[24];
  x(0, 2, 1, 1) = x(2, 0, 1, 1) = input[25];
  x(0, 2, 2, 2) = x(2, 0, 2, 2) = input[26];
  x(0, 2, 1, 2) = x(0, 2, 2, 1) = x(2, 0, 1, 2) = x(2, 0, 2, 1) = input[27];
  x(0, 2, 0, 2) = x(0, 2, 2, 0) = x(2, 0, 0, 2) = x(2, 0, 2, 0) = input[28];
  x(0, 2, 0, 1) = x(0, 2, 1, 0) = x(2, 0, 0, 1) = x(2, 0, 1, 0) = input[29];

  x(0, 1, 0, 0) = x(1, 0, 0, 0) = input[30];
  x(0, 1, 1, 1) = x(1, 0, 1, 1) = input[31];
  x(0, 1, 2, 2) = x(1, 0, 2, 2) = input[32];
  x(0, 1, 1, 2) = x(0, 1, 2, 1) = x(1, 0, 1, 2) = x(1, 0, 2, 1) = input[33];
  x(0, 1, 0, 2) = x(0, 1, 2, 0) = x(1, 0, 0, 2) = x(1, 0, 2, 0) = input[34];
  x(0, 1, 0, 1) = x(0, 1, 1, 0) = x(1, 0, 0, 1) = x(1, 0, 1, 0) = input[35];
}

#ifdef LIBTORCH_ENABLED

void
fillSymmetricRankFourFromInputTensor(RankFourTensor & x,
                                     const torch::Tensor & input,
                                     const bool isEngineeringStrain)
{
  const Real multiplier = isEngineeringStrain ? 0.5 : 1.0;
  if (input.sizes() == torch::IntList({36}))
  {
    // shear strains are kept original.
    x(0, 0, 0, 0) = input[0].item<double>();
    x(0, 0, 1, 1) = input[1].item<double>();
    x(0, 0, 2, 2) = input[2].item<double>();
    x(0, 0, 1, 2) = x(0, 0, 2, 1) = input[3].item<double>() * multiplier;
    x(0, 0, 0, 2) = x(0, 0, 2, 0) = input[4].item<double>() * multiplier;
    x(0, 0, 0, 1) = x(0, 0, 1, 0) = input[5].item<double>() * multiplier;

    x(1, 1, 0, 0) = input[6].item<double>();
    x(1, 1, 1, 1) = input[7].item<double>();
    x(1, 1, 2, 2) = input[8].item<double>();
    x(1, 1, 1, 2) = x(1, 1, 2, 1) = input[9].item<double>() * multiplier;
    x(1, 1, 0, 2) = x(1, 1, 2, 0) = input[10].item<double>() * multiplier;
    x(1, 1, 0, 1) = x(1, 1, 1, 0) = input[11].item<double>() * multiplier;

    x(2, 2, 0, 0) = input[12].item<double>();
    x(2, 2, 1, 1) = input[13].item<double>();
    x(2, 2, 2, 2) = input[14].item<double>();
    x(2, 2, 1, 2) = x(2, 2, 2, 1) = input[15].item<double>() * multiplier;
    x(2, 2, 0, 2) = x(2, 2, 2, 0) = input[16].item<double>() * multiplier;
    x(2, 2, 0, 1) = x(2, 2, 1, 0) = input[17].item<double>() * multiplier;

    x(1, 2, 0, 0) = x(2, 1, 0, 0) = input[18].item<double>();
    x(1, 2, 1, 1) = x(2, 1, 1, 1) = input[19].item<double>();
    x(1, 2, 2, 2) = x(2, 1, 2, 2) = input[20].item<double>();
    x(1, 2, 1, 2) = x(1, 2, 2, 1) = x(2, 1, 1, 2) = x(2, 1, 2, 1) =
        input[21].item<double>() * multiplier;
    x(1, 2, 0, 2) = x(1, 2, 2, 0) = x(2, 1, 0, 2) = x(2, 1, 2, 0) =
        input[22].item<double>() * multiplier;
    x(1, 2, 0, 1) = x(1, 2, 1, 0) = x(2, 1, 0, 1) = x(2, 1, 1, 0) =
        input[23].item<double>() * multiplier;

    x(0, 2, 0, 0) = x(2, 0, 0, 0) = input[24].item<double>();
    x(0, 2, 1, 1) = x(2, 0, 1, 1) = input[25].item<double>();
    x(0, 2, 2, 2) = x(2, 0, 2, 2) = input[26].item<double>();
    x(0, 2, 1, 2) = x(0, 2, 2, 1) = x(2, 0, 1, 2) = x(2, 0, 2, 1) =
        input[27].item<double>() * multiplier;
    x(0, 2, 0, 2) = x(0, 2, 2, 0) = x(2, 0, 0, 2) = x(2, 0, 2, 0) =
        input[28].item<double>() * multiplier;
    x(0, 2, 0, 1) = x(0, 2, 1, 0) = x(2, 0, 0, 1) = x(2, 0, 1, 0) =
        input[29].item<double>() * multiplier;

    x(0, 1, 0, 0) = x(1, 0, 0, 0) = input[30].item<double>();
    x(0, 1, 1, 1) = x(1, 0, 1, 1) = input[31].item<double>();
    x(0, 1, 2, 2) = x(1, 0, 2, 2) = input[32].item<double>();
    x(0, 1, 1, 2) = x(0, 1, 2, 1) = x(1, 0, 1, 2) = x(1, 0, 2, 1) =
        input[33].item<double>() * multiplier;
    x(0, 1, 0, 2) = x(0, 1, 2, 0) = x(1, 0, 0, 2) = x(1, 0, 2, 0) =
        input[34].item<double>() * multiplier;
    x(0, 1, 0, 1) = x(0, 1, 1, 0) = x(1, 0, 0, 1) = x(1, 0, 1, 0) =
        input[35].item<double>() * multiplier;
  }
  else if (input.sizes() == torch::IntList({9}))
  {
    // shear strains are kept original.
    x(0, 0, 0, 0) = input[0].item<double>();
    x(0, 0, 1, 1) = input[1].item<double>();
    x(0, 0, 0, 1) = x(0, 0, 1, 0) = input[2].item<double>() * multiplier;

    x(1, 1, 0, 0) = input[3].item<double>();
    x(1, 1, 1, 1) = input[4].item<double>();
    x(1, 1, 0, 1) = x(1, 1, 1, 0) = input[5].item<double>() * multiplier;

    x(0, 1, 0, 0) = x(1, 0, 0, 0) = input[6].item<double>();
    x(0, 1, 1, 1) = x(1, 0, 1, 1) = input[7].item<double>();
    x(0, 1, 0, 1) = x(0, 1, 1, 0) = x(1, 0, 0, 1) = x(1, 0, 1, 0) =
        input[8].item<double>() * multiplier;
  }
  else
  {
    mooseError("Input tensor must be of size 36 or 9 for symmetric rank 4 tensor.\n");
  }
}

void
fillRankTwoFromInputTensor(RankTwoTensor & x, const torch::Tensor & input)
{
  int dim;
  if (input.size(0) == 9)
    dim = 3;
  else if (input.size(0) == 4)
    dim = 2;
  else
    mooseError("stress tensor's size must be 9 or 4.\n");

  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = 0; j < dim; j++)
      x(i, j) = input[i * dim + j].item<double>();
}

void
fillRankFourFromInputTensor(RankFourTensor & x, const torch::Tensor & input)
{
  int dim;
  if (input.size(0) == 81)
    dim = 3;
  else if (input.size(0) == 16)
    dim = 2;
  else
    mooseError("Jacobian tensor's size must be 81 or 16.\n");

  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = 0; j < dim; j++)
      for (unsigned int k = 0; k < dim; k++)
        for (unsigned int l = 0; l < dim; l++)
          x(i, j, k, l) = input[i * dim * dim * dim + j * dim * dim + k * dim + l].item<double>();
}

void
fillSymmeticRankTwoFromInputTensor(RankTwoTensor & x, const torch::Tensor & input)
{
  if (input.sizes() == torch::IntList({6}))
  {
    x(0, 0) = input[0].item<double>();
    x(1, 1) = input[1].item<double>();
    x(2, 2) = input[2].item<double>();
    x(1, 2) = x(2, 1) = input[3].item<double>();
    x(0, 2) = x(2, 0) = input[4].item<double>();
    x(0, 1) = x(1, 0) = input[5].item<double>();
  }
  else if (input.sizes() == torch::IntList({3}))
  {
    x(0, 0) = input[0].item<double>();
    x(1, 1) = input[1].item<double>();
    x(0, 1) = x(1, 0) = input[2].item<double>();
  }
  else
  {
    mooseError("Input tensor must be of size 6 or 3 for symmetric rank 2 tensor.\n");
  }
}

RankFourTensor
SymmetricRankFourTensorFromTensor(const torch::Tensor & input, unsigned int dim)
{
  mooseAssert(input.dim() == 1, "input dim must be 1.");
  RankFourTensor t;

  if (dim == 3)
  {
    t(0, 0, 0, 0) = input[0].item<Real>();  // C1111
    t(1, 1, 1, 1) = input[6].item<Real>();  // C2222
    t(2, 2, 2, 2) = input[11].item<Real>(); // C3333

    t(0, 0, 1, 1) = input[1].item<Real>(); // C1122
    t(1, 1, 0, 0) = input[1].item<Real>();

    t(0, 0, 2, 2) = input[2].item<Real>(); // C1133
    t(2, 2, 0, 0) = input[2].item<Real>();

    t(1, 1, 2, 2) = input[7].item<Real>(); // C2233
    t(2, 2, 1, 1) = input[7].item<Real>();

    t(0, 0, 0, 2) = input[4].item<Real>(); // C1113
    t(0, 0, 2, 0) = input[4].item<Real>();
    t(0, 2, 0, 0) = input[4].item<Real>();
    t(2, 0, 0, 0) = input[4].item<Real>();

    t(0, 0, 0, 1) = input[5].item<Real>(); // C1112
    t(0, 0, 1, 0) = input[5].item<Real>();
    t(0, 1, 0, 0) = input[5].item<Real>();
    t(1, 0, 0, 0) = input[5].item<Real>();

    t(1, 1, 1, 2) = input[8].item<Real>(); // C2223
    t(1, 1, 2, 1) = input[8].item<Real>();
    t(1, 2, 1, 1) = input[8].item<Real>();
    t(2, 1, 1, 1) = input[8].item<Real>();

    t(1, 1, 1, 0) = input[10].item<Real>();
    t(1, 1, 0, 1) = input[10].item<Real>();
    t(1, 0, 1, 1) = input[10].item<Real>();
    t(0, 1, 1, 1) = input[10].item<Real>(); // C2212 //flipped for filling purposes

    t(2, 2, 2, 1) = input[12].item<Real>();
    t(2, 2, 1, 2) = input[12].item<Real>();
    t(2, 1, 2, 2) = input[12].item<Real>();
    t(1, 2, 2, 2) = input[12].item<Real>(); // C3323 //flipped for filling purposes

    t(2, 2, 2, 0) = input[13].item<Real>();
    t(2, 2, 0, 2) = input[13].item<Real>();
    t(2, 0, 2, 2) = input[13].item<Real>();
    t(0, 2, 2, 2) = input[13].item<Real>(); // C3313 //flipped for filling purposes

    t(0, 0, 1, 2) = input[3].item<Real>(); // C1123
    t(0, 0, 2, 1) = input[3].item<Real>();
    t(1, 2, 0, 0) = input[3].item<Real>();
    t(2, 1, 0, 0) = input[3].item<Real>();

    t(1, 1, 0, 2) = input[9].item<Real>();
    t(1, 1, 2, 0) = input[9].item<Real>();
    t(0, 2, 1, 1) = input[9].item<Real>(); // C2213  //flipped for filling purposes
    t(2, 0, 1, 1) = input[9].item<Real>();

    t(2, 2, 0, 1) = input[14].item<Real>();
    t(2, 2, 1, 0) = input[14].item<Real>();
    t(0, 1, 2, 2) = input[14].item<Real>(); // C3312 //flipped for filling purposes
    t(1, 0, 2, 2) = input[14].item<Real>();

    t(1, 2, 1, 2) = input[15].item<Real>(); // C2323
    t(2, 1, 2, 1) = input[15].item<Real>();
    t(2, 1, 1, 2) = input[15].item<Real>();
    t(1, 2, 2, 1) = input[15].item<Real>();

    t(0, 2, 0, 2) = input[18].item<Real>(); // C1313
    t(2, 0, 2, 0) = input[18].item<Real>();
    t(2, 0, 0, 2) = input[18].item<Real>();
    t(0, 2, 2, 0) = input[18].item<Real>();

    t(0, 1, 0, 1) = input[20].item<Real>(); // C1212
    t(1, 0, 1, 0) = input[20].item<Real>();
    t(1, 0, 0, 1) = input[20].item<Real>();
    t(0, 1, 1, 0) = input[20].item<Real>();

    t(1, 2, 0, 2) = input[16].item<Real>();
    t(0, 2, 1, 2) = input[16].item<Real>(); // C2313 //flipped for filling purposes
    t(2, 1, 0, 2) = input[16].item<Real>();
    t(1, 2, 2, 0) = input[16].item<Real>();
    t(2, 0, 1, 2) = input[16].item<Real>();
    t(0, 2, 2, 1) = input[16].item<Real>();
    t(2, 1, 2, 0) = input[16].item<Real>();
    t(2, 0, 2, 1) = input[16].item<Real>();

    t(1, 2, 0, 1) = input[17].item<Real>();
    t(0, 1, 1, 2) = input[17].item<Real>(); // C2312 //flipped for filling purposes
    t(2, 1, 0, 1) = input[17].item<Real>();
    t(1, 2, 1, 0) = input[17].item<Real>();
    t(1, 0, 1, 2) = input[17].item<Real>();
    t(0, 1, 2, 1) = input[17].item<Real>();
    t(2, 1, 1, 0) = input[17].item<Real>();
    t(1, 0, 2, 1) = input[17].item<Real>();

    t(0, 2, 0, 1) = input[19].item<Real>();
    t(0, 1, 0, 2) = input[19].item<Real>(); // C1312 //flipped for filling purposes
    t(2, 0, 0, 1) = input[19].item<Real>();
    t(0, 2, 1, 0) = input[19].item<Real>();
    t(1, 0, 0, 2) = input[19].item<Real>();
    t(0, 1, 2, 0) = input[19].item<Real>();
    t(2, 0, 1, 0) = input[19].item<Real>();
    t(1, 0, 2, 0) = input[19].item<Real>();
  }
  else
  {
    t(0, 0, 0, 0) = input[0].item<Real>(); // C1111
    t(1, 1, 1, 1) = input[3].item<Real>(); // C2222

    t(0, 0, 1, 1) = input[1].item<Real>(); // C1122
    t(1, 1, 0, 0) = input[1].item<Real>();

    t(0, 0, 0, 1) = input[2].item<Real>(); // C1112
    t(0, 0, 1, 0) = input[2].item<Real>();
    t(0, 1, 0, 0) = input[2].item<Real>();
    t(1, 0, 0, 0) = input[2].item<Real>();

    t(1, 1, 1, 0) = input[4].item<Real>();
    t(1, 1, 0, 1) = input[4].item<Real>();
    t(1, 0, 1, 1) = input[4].item<Real>();
    t(0, 1, 1, 1) = input[4].item<Real>(); // C2212 //flipped for filling purposes

    t(0, 1, 0, 1) = input[5].item<Real>(); // C1212
    t(1, 0, 1, 0) = input[5].item<Real>();
    t(1, 0, 0, 1) = input[5].item<Real>();
    t(0, 1, 1, 0) = input[5].item<Real>();
  }
  return t;
}

#endif

void
dofMap::insert(unsigned int node_id, std::vector<unsigned int> dof_ids)
{
  // erase old node
  if (_dof_ids.find(node_id) != _dof_ids.end())
  {
    auto old_dof_ids = _dof_ids[node_id];
    for (auto && old_dof_id : old_dof_ids)
    {
      _node_id.erase(old_dof_id);
    }
  }

  for (auto && dof_id : dof_ids)
  {
    // check if dof occupied
    if (_node_id.find(dof_id) != _node_id.end())
    {
      mooseError("node ", node_id, ": dof ", dof_id, " occupied!");
    }
    _node_id[dof_id] = node_id;
  }

  _dof_ids[node_id] = dof_ids;
}

std::vector<unsigned int>
dofMap::getDofIds(unsigned int node_id) const
{
  auto it = _dof_ids.find(node_id);
  if (it != _dof_ids.end())
  {
    return it->second;
  }
  throw std::out_of_range("Required dof map does not exist!");
}

unsigned int
dofMap::getNodeId(unsigned int dof_id) const
{
  auto it = _node_id.find(dof_id);
  if (it != _node_id.end())
  {
    return it->second;
  }
  throw std::out_of_range("Required dof map does not exist!");
}

void
dofMap::clear()
{
  _dof_ids.clear();
  _node_id.clear();
}

void
dofMap::print(std::ostream & os)
{
  os << "Node_id\tDof_ids\n";
  for (auto && m : _dof_ids)
  {
    os << m.first << "\t";
    auto dof_ids = m.second;
    os << "{";
    bool isfirst = true;
    for (auto && dof_id : dof_ids)
    {
      if (!isfirst)
        os << " ";
      os << dof_id;
      isfirst = false;
    }
    os << "}\n";
  }
  os << std::endl;
}

void
nodeMap::insert(unsigned int node_id, unsigned int map_id)
{
  // erase old node
  if (_map_id.find(node_id) != _map_id.end())
  {
    auto old_map_id = _map_id[node_id];
    _node_id.erase(old_map_id);
  }

  if (_node_id.find(map_id) != _node_id.end())
  {
    mooseError("node ", node_id, ": map ", map_id, " occupied!");
  }

  _node_id[map_id] = node_id;
  _map_id[node_id] = map_id;
}

unsigned int
nodeMap::getMapId(unsigned int node_id) const
{
  auto it = _map_id.find(node_id);
  if (it != _map_id.end())
  {
    return it->second;
  }
  throw std::out_of_range("Required map does not exist!");
}

unsigned int
nodeMap::getNodeId(unsigned int map_id) const
{
  auto it = _node_id.find(map_id);
  if (it != _node_id.end())
  {
    return it->second;
  }
  throw std::out_of_range("Required node map does not exist!");
}

bool
nodeMap::hasNodeId(unsigned int node_id) const
{
  auto it = _map_id.find(node_id);
  if (it != _map_id.end())
    return true;
  return false;
}

void
nodeMap::clear()
{
  _map_id.clear();
  _node_id.clear();
}

void
nodeMap::print(std::ostream & os)
{
  os << "Node_id\tMap_id\n";
  for (auto && m : _map_id)
  {
    os << m.first << "\t" << m.second << "\n";
  }
  os << std::endl;
}

RankFourTensor
SymmetricRankFourTensorFromInputVector(const std::vector<Real> & input, unsigned int dim)
{
  mooseAssert((input.size() == 21 && dim == 3) || (input.size() == 6 && dim == 2),
              "Input must have size 21(3d) or 6(2d).");

  RankFourTensor t;

  if (dim == 3)
  {
    t(0, 0, 0, 0) = input[0];  // C1111
    t(1, 1, 1, 1) = input[6];  // C2222
    t(2, 2, 2, 2) = input[11]; // C3333

    t(0, 0, 1, 1) = input[1]; // C1122
    t(1, 1, 0, 0) = input[1];

    t(0, 0, 2, 2) = input[2]; // C1133
    t(2, 2, 0, 0) = input[2];

    t(1, 1, 2, 2) = input[7]; // C2233
    t(2, 2, 1, 1) = input[7];

    t(0, 0, 0, 2) = input[4]; // C1113
    t(0, 0, 2, 0) = input[4];
    t(0, 2, 0, 0) = input[4];
    t(2, 0, 0, 0) = input[4];

    t(0, 0, 0, 1) = input[5]; // C1112
    t(0, 0, 1, 0) = input[5];
    t(0, 1, 0, 0) = input[5];
    t(1, 0, 0, 0) = input[5];

    t(1, 1, 1, 2) = input[8]; // C2223
    t(1, 1, 2, 1) = input[8];
    t(1, 2, 1, 1) = input[8];
    t(2, 1, 1, 1) = input[8];

    t(1, 1, 1, 0) = input[10];
    t(1, 1, 0, 1) = input[10];
    t(1, 0, 1, 1) = input[10];
    t(0, 1, 1, 1) = input[10]; // C2212 //flipped for filling purposes

    t(2, 2, 2, 1) = input[12];
    t(2, 2, 1, 2) = input[12];
    t(2, 1, 2, 2) = input[12];
    t(1, 2, 2, 2) = input[12]; // C3323 //flipped for filling purposes

    t(2, 2, 2, 0) = input[13];
    t(2, 2, 0, 2) = input[13];
    t(2, 0, 2, 2) = input[13];
    t(0, 2, 2, 2) = input[13]; // C3313 //flipped for filling purposes

    t(0, 0, 1, 2) = input[3]; // C1123
    t(0, 0, 2, 1) = input[3];
    t(1, 2, 0, 0) = input[3];
    t(2, 1, 0, 0) = input[3];

    t(1, 1, 0, 2) = input[9];
    t(1, 1, 2, 0) = input[9];
    t(0, 2, 1, 1) = input[9]; // C2213  //flipped for filling purposes
    t(2, 0, 1, 1) = input[9];

    t(2, 2, 0, 1) = input[14];
    t(2, 2, 1, 0) = input[14];
    t(0, 1, 2, 2) = input[14]; // C3312 //flipped for filling purposes
    t(1, 0, 2, 2) = input[14];

    t(1, 2, 1, 2) = input[15]; // C2323
    t(2, 1, 2, 1) = input[15];
    t(2, 1, 1, 2) = input[15];
    t(1, 2, 2, 1) = input[15];

    t(0, 2, 0, 2) = input[18]; // C1313
    t(2, 0, 2, 0) = input[18];
    t(2, 0, 0, 2) = input[18];
    t(0, 2, 2, 0) = input[18];

    t(0, 1, 0, 1) = input[20]; // C1212
    t(1, 0, 1, 0) = input[20];
    t(1, 0, 0, 1) = input[20];
    t(0, 1, 1, 0) = input[20];

    t(1, 2, 0, 2) = input[16];
    t(0, 2, 1, 2) = input[16]; // C2313 //flipped for filling purposes
    t(2, 1, 0, 2) = input[16];
    t(1, 2, 2, 0) = input[16];
    t(2, 0, 1, 2) = input[16];
    t(0, 2, 2, 1) = input[16];
    t(2, 1, 2, 0) = input[16];
    t(2, 0, 2, 1) = input[16];

    t(1, 2, 0, 1) = input[17];
    t(0, 1, 1, 2) = input[17]; // C2312 //flipped for filling purposes
    t(2, 1, 0, 1) = input[17];
    t(1, 2, 1, 0) = input[17];
    t(1, 0, 1, 2) = input[17];
    t(0, 1, 2, 1) = input[17];
    t(2, 1, 1, 0) = input[17];
    t(1, 0, 2, 1) = input[17];

    t(0, 2, 0, 1) = input[19];
    t(0, 1, 0, 2) = input[19]; // C1312 //flipped for filling purposes
    t(2, 0, 0, 1) = input[19];
    t(0, 2, 1, 0) = input[19];
    t(1, 0, 0, 2) = input[19];
    t(0, 1, 2, 0) = input[19];
    t(2, 0, 1, 0) = input[19];
    t(1, 0, 2, 0) = input[19];
  }
  else
  {
    t(0, 0, 0, 0) = input[0]; // C1111
    t(1, 1, 1, 1) = input[3]; // C2222

    t(0, 0, 1, 1) = input[1]; // C1122
    t(1, 1, 0, 0) = input[1];

    t(0, 0, 0, 1) = input[2]; // C1112
    t(0, 0, 1, 0) = input[2];
    t(0, 1, 0, 0) = input[2];
    t(1, 0, 0, 0) = input[2];

    t(1, 1, 1, 0) = input[4];
    t(1, 1, 0, 1) = input[4];
    t(1, 0, 1, 1) = input[4];
    t(0, 1, 1, 1) = input[4]; // C2212 //flipped for filling purposes

    t(0, 1, 0, 1) = input[5]; // C1212
    t(1, 0, 1, 0) = input[5];
    t(1, 0, 0, 1) = input[5];
    t(0, 1, 1, 0) = input[5];
  }
  return t;
}

std::vector<Real>
VectorFromInputSymmetricTensor(const RankFourTensor & input, unsigned int dim)
{
  mooseAssert(dim == 3 || dim == 2, "Unsupported dim.");
  std::vector<Real> v;
  if (dim == 3)
  {
    v.resize(21);

    v[0] = input(0, 0, 0, 0);
    v[1] = input(0, 0, 1, 1);
    v[2] = input(0, 0, 2, 2);
    v[3] = input(0, 0, 1, 2);
    v[4] = input(0, 0, 0, 2);
    v[5] = input(0, 0, 0, 1);
    v[6] = input(1, 1, 1, 1);
    v[7] = input(1, 1, 2, 2);
    v[8] = input(1, 1, 1, 2);
    v[9] = input(1, 1, 0, 2);
    v[10] = input(1, 1, 0, 1);
    v[11] = input(2, 2, 2, 2);
    v[12] = input(2, 2, 1, 2);
    v[13] = input(2, 2, 0, 2);
    v[14] = input(2, 2, 0, 1);
    v[15] = input(1, 2, 1, 2);
    v[16] = input(1, 2, 0, 2);
    v[17] = input(1, 2, 0, 1);
    v[18] = input(0, 2, 0, 2);
    v[19] = input(0, 2, 0, 1);
    v[20] = input(0, 1, 0, 1);
  }
  else if (dim == 2)
  {
    v.resize(6);

    v[0] = input(0, 0, 0, 0);
    v[1] = input(0, 0, 1, 1);
    v[2] = input(0, 0, 0, 1);
    v[3] = input(1, 1, 1, 1);
    v[4] = input(1, 1, 0, 1);
    v[5] = input(0, 1, 0, 1);
  }
  return v;
}

std::vector<Real>
VectorFromInputSymmetricTensor(const RankTwoTensor & input, unsigned int dim)
{
  mooseAssert(dim == 3 || dim == 2, "Unsupported dim.");
  std::vector<Real> v;
  if (dim == 3)
  {
    v.resize(6);

    v[0] = input(0, 0);
    v[1] = input(1, 1);
    v[2] = input(2, 2);
    v[3] = input(1, 2);
    v[4] = input(0, 2);
    v[5] = input(0, 1);
  }
  else if (dim == 2)
  {
    v.resize(3);

    v[0] = input(0, 0);
    v[1] = input(1, 1);
    v[2] = input(0, 1);
  }
  return v;
}
}

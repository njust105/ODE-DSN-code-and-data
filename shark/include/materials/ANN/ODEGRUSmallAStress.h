#ifdef LIBTORCH_ENABLED

#pragma once
#include "MooseUtils.h"
#include <torch/torch.h>
#include "nlohmann/json.h"
#include "ComputeLagrangianObjectiveStress.h"
#include "GRUaStress.h"

class ODEGRUSmallAStress : public ComputeLagrangianObjectiveStress
{
public:
  static InputParameters validParams();
  ODEGRUSmallAStress(const InputParameters & parameters);

protected:
  virtual void computeQpCauchyStress() override;
  virtual void computeQpSmallStress() override;

  torch::Tensor rotatedDeformationTensor(const RankTwoTensor & x, const RankTwoTensor & R);

  std::string _model_file;
  std::string _configure_file;
  const bool _use_ad;

  const MaterialProperty<RankTwoTensor> & _mechanical_strain_old;

  MaterialProperty<std::vector<Real>> & _hidden_state;
  const MaterialProperty<std::vector<Real>> & _hidden_state_old;

  const MaterialProperty<RankTwoTensor> & _rotation_matrix;

  GRUaStress _gru_stress;
  nlohmann::json _config;

private:
  unsigned int _dim;

  unsigned int _gru_layers;
  unsigned int _hidden_size;
  Real _E_scaling;
  Real _S_scaling;
};

#endif
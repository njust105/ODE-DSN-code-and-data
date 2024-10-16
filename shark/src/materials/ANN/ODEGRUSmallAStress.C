#ifdef LIBTORCH_ENABLED

#include "ODEGRUSmallAStress.h"
#include "SharkUtils.h"
#include <torch/torch.h>

registerMooseObject("sharkApp", ODEGRUSmallAStress);

InputParameters
ODEGRUSmallAStress::validParams()
{
  InputParameters params = ComputeLagrangianObjectiveStress::validParams();
  params.addRequiredParam<std::string>("model", "file name of the torch model");
  params.addRequiredParam<std::string>("configure", "file name of the configure file");
  params.addRequiredParam<MaterialPropertyName>("rotation_matrix",
                                                "Material name of the rotation matrix.");
  params.addParam<bool>("use_ad", true, "use autograd.");

  return params;
}

ODEGRUSmallAStress::ODEGRUSmallAStress(const InputParameters & parameters)
  : ComputeLagrangianObjectiveStress(parameters),
    _model_file(getParam<std::string>("model")),
    _configure_file(getParam<std::string>("configure")),
    _use_ad(getParam<bool>("use_ad")),
    _mechanical_strain_old(getMaterialPropertyOld<RankTwoTensor>(_base_name + "mechanical_strain")),
    _hidden_state(declareProperty<std::vector<Real>>(_base_name + "hidden_state")),
    _hidden_state_old(getMaterialPropertyOld<std::vector<Real>>(_base_name + "hidden_state")),
    _rotation_matrix(getMaterialProperty<RankTwoTensor>("rotation_matrix")),
    _gru_stress(nullptr)
{
  try
  {
    std::ifstream configure(_configure_file);
    configure >> _config;
    configure.close();
  }
  catch (const std::exception & e)
  {
    mooseError("Error while loading configure. ", _configure_file, "!\n", e.what());
  }

  try
  {
    torch::cuda::is_available();
    _gru_stress = GRUaStress(_config["model"]);
    _gru_stress->to(torch::kFloat64);
    torch::load(_gru_stress, _model_file);
    _gru_stress->eval();
  }
  catch (const c10::Error & e)
  {
    mooseError("Error while creating torch model ", _model_file, "!\n", e.msg());
  }

  _dim = _config["model"]["input_output"]["dim"];
  _S_scaling = _config["scaling"]["S"];
  _E_scaling = _config["scaling"]["E"];
  _hidden_size = _config["model"]["gru"]["hidden_size"];
  _gru_layers = _config["model"]["gru"]["num_layers"];
}

void
ODEGRUSmallAStress::computeQpCauchyStress()
{
  if (_use_ad && _fe_problem.currentlyComputingJacobian())
  {
    ComputeLagrangianObjectiveStress::computeQpCauchyStress();
  }
  else
  {
    torch::NoGradGuard no_grad;
    ComputeLagrangianObjectiveStress::computeQpCauchyStress();
  }
}

void
ODEGRUSmallAStress::computeQpSmallStress()
{
  torch::Tensor deformation =
      rotatedDeformationTensor(_mechanical_strain[_qp], _rotation_matrix[_qp]) / _E_scaling;

  torch::Tensor deformation_old =
      rotatedDeformationTensor(_mechanical_strain_old[_qp], _rotation_matrix[_qp])
          .unsqueeze(0)
          .unsqueeze(0) /
      _E_scaling;

  torch::Tensor stress_old = rotatedDeformationTensor(_small_stress_old[_qp], _rotation_matrix[_qp])
                                 .unsqueeze(0)
                                 .unsqueeze(0) /
                             _S_scaling;

  torch::Tensor hidden_state_old, hidden_state, stress_and_jacobian;
  if (_t_step >= 2)
  {
    hidden_state_old = torch::from_blob(const_cast<Real *>(_hidden_state_old[_qp].data()),
                                        {_gru_layers, 1, _hidden_size},
                                        torch::kFloat64);
  }

  // use autograd to solve the jacobian
  if (_use_ad && _fe_problem.currentlyComputingJacobian())
    deformation.requires_grad_();

  torch::Tensor tt = torch::tensor({_t}, deformation.options());
  torch::Tensor tt_old = torch::tensor({{_t - _dt}}, deformation.options());

  torch::Tensor timestamped_deformation = torch::cat({tt, deformation}, -1);
  timestamped_deformation.unsqueeze_(0);
  timestamped_deformation.unsqueeze_(0);

  std::tie(stress_and_jacobian, hidden_state) = _gru_stress->forward(
      timestamped_deformation, deformation_old, stress_old, tt_old, hidden_state_old);

  torch::Tensor stress = stress_and_jacobian.slice(-1, 0, deformation.size(-1)).view({-1});
  torch::Tensor jacobian = stress_and_jacobian.slice(-1, deformation.size(-1)).view({-1});

  SharkUtils::fillVectorFromTensor(_hidden_state[_qp], hidden_state);
  if (_fe_problem.currentlyComputingJacobian())
  {
    if (_use_ad)
    {
      std::vector<torch::Tensor> grads;
      for (int component = 0; component < (1 + _dim) * _dim / 2; component++)
      {
        torch::Tensor grad = torch::autograd::grad({stress[component]}, {deformation}, {}, true)[0];
        grads.push_back(grad);
      }
      torch::Tensor concatenated_grads = torch::cat(grads, 0);
      SharkUtils::fillSymmetricRankFourFromInputTensor(_small_jacobian[_qp], concatenated_grads);
    }
    else
    {
      bool sym = _config["model"]["input_output"]["jacobian_sym"];
      if (sym)
      {
        _small_jacobian[_qp] = SharkUtils::SymmetricRankFourTensorFromTensor(jacobian, _dim);
      }
      else
      {
        SharkUtils::fillSymmetricRankFourFromInputTensor(_small_jacobian[_qp], jacobian, false);
      }
    }
    _small_jacobian[_qp] *= _S_scaling / _E_scaling;

    _small_jacobian[_qp].rotate(_rotation_matrix[_qp]);
    RankFourTensor Iikjl(RankFourTensor::InitMethod::initIdentityFour);
    RankTwoTensor I2(RankTwoTensor::InitMethod::initIdentity);
    //  0.5 * （delta_ik delta_jl + delta_jk delta_il)
    _small_jacobian[_qp] = _small_jacobian[_qp] * (Iikjl + I2.times<0, 3, 1, 2>(I2)) / 2;
  }
  SharkUtils::fillSymmeticRankTwoFromInputTensor(_small_stress[_qp],
                                                 (stress * _S_scaling).view({-1}));

  _small_stress[_qp].rotate(_rotation_matrix[_qp]);
}

torch::Tensor
ODEGRUSmallAStress::rotatedDeformationTensor(const RankTwoTensor & x, const RankTwoTensor & R)
{
  RankTwoTensor _x = x.rotated(R.transpose());
  switch (_dim)
  {
    case 2:
      return torch::tensor({_x(0, 0), _x(1, 1), _x(0, 1)}, torch::kFloat64);
    case 3:
      return torch::tensor({_x(0, 0), _x(1, 1), _x(2, 2), _x(1, 2), _x(0, 2), _x(0, 1)},
                           torch::kFloat64);
    default:
      mooseError("Wrong dim.\n");
  }
  return torch::Tensor();
}

#endif
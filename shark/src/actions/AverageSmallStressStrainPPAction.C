#include "AverageSmallStressStrainPPAction.h"
#include "Factory.h"
#include "FEProblem.h"
#include "Conversion.h"

registerMooseAction("sharkApp", AverageSmallStressStrainPPAction, "add_postprocessor");

InputParameters
AverageSmallStressStrainPPAction::validParams()
{
  InputParameters params = Action::validParams();
  params.addClassDescription("Set up postprocessors for PK1 stress and deformation gradient");

  params.addParam<bool>("use_displaced_mesh", false, "Whether to use the displaced mesh.");

  params.addParam<std::string>("base_name", "The base name of the material");
  params.addParam<unsigned int>("dim", 3, "Dim");
  params.addParam<std::string>(
      "strain_pp_base_name", "E", "The base name of strain postprocessors.");
  params.addParam<std::string>(
      "stress_pp_base_name", "S", "The base name of stress postprocessors.");
  return params;
}

AverageSmallStressStrainPPAction::AverageSmallStressStrainPPAction(const InputParameters & params)
  : Action(params),
    _base_name(isParamValid("base_name") ? getParam<std::string>("base_name") + "_" : ""),
    _dim(getParam<unsigned int>("dim")),
    _strain_pp_base_name(getParam<std::string>("strain_pp_base_name")),
    _stress_pp_base_name(getParam<std::string>("stress_pp_base_name"))
{
}

void
AverageSmallStressStrainPPAction::act()
{
  if (_current_task == "add_postprocessor")
  {
    for (unsigned int j = 0; j < _dim; j++)
    {
      for (unsigned int i = 0; i <= j; i++)
      {
        std::string strain_name =
            _strain_pp_base_name + std::to_string(i + 1) + std::to_string(j + 1);
        InputParameters params_strain = _factory.getValidParams("MaterialTensorAverage");
        params_strain.set<unsigned int>("index_i") = i;
        params_strain.set<unsigned int>("index_j") = j;
        params_strain.set<MaterialPropertyName>("rank_two_tensor") =
            _base_name + "mechanical_strain";
        params_strain.set<bool>("use_displaced_mesh") = getParam<bool>("use_displaced_mesh");
        _problem->addPostprocessor("MaterialTensorAverage", strain_name, params_strain);

        std::string stress_name =
            _stress_pp_base_name + std::to_string(i + 1) + std::to_string(j + 1);
        InputParameters params_stress = _factory.getValidParams("MaterialTensorAverage");
        params_stress.set<unsigned int>("index_i") = i;
        params_stress.set<unsigned int>("index_j") = j;
        params_stress.set<MaterialPropertyName>("rank_two_tensor") = _base_name + "small_stress";
        params_stress.set<bool>("use_displaced_mesh") = getParam<bool>("use_displaced_mesh");
        _problem->addPostprocessor("MaterialTensorAverage", stress_name, params_stress);
      }
    }
  }
}

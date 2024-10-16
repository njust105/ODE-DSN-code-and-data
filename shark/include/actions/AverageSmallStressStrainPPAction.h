
#pragma once

#include "Action.h"

class AverageSmallStressStrainPPAction : public Action
{
public:
  static InputParameters validParams();

  AverageSmallStressStrainPPAction(const InputParameters & params);

  virtual void act() override;

protected:
  std::string _base_name;
  unsigned int _dim;
  std::string _strain_pp_base_name;
  std::string _stress_pp_base_name;
};

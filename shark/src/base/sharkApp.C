#include "sharkApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
sharkApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_bahavior") = false;
  return params;
}

sharkApp::sharkApp(InputParameters parameters) : MooseApp(parameters)
{
  sharkApp::registerAll(_factory, _action_factory, _syntax);
}

sharkApp::~sharkApp() {}

static void
associateSyntaxInner(Syntax & syntax, ActionFactory & /*action_factory*/)
{
  registerSyntax("EmptyAction", "Postprocessors/AverageSmallStressStrain");
  registerSyntax("AverageSmallStressStrainPPAction", "Postprocessors/AverageSmallStressStrain/*");
}

void
sharkApp::registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  ModulesApp::registerAllObjects<sharkApp>(f, af, s);
  Registry::registerObjectsTo(f, {"sharkApp"});
  Registry::registerActionsTo(af, {"sharkApp"});

  /* register custom execute flags, action syntax, etc. here */
  associateSyntaxInner(s, af);
}

void
sharkApp::registerApps()
{
  registerApp(sharkApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
sharkApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  sharkApp::registerAll(f, af, s);
}
extern "C" void
sharkApp__registerApps()
{
  sharkApp::registerApps();
}

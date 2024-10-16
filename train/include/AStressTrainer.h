#pragma once

#include "Trainer.h"
#include "GRUaStress.h"

template <typename LossFunctionType, typename DataLoaderType1, typename DataLoaderType2, typename OptimizerType, typename LRSchedulertype>
class AStressTrainer : public Trainer<GRUaStress, LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>
{
public:
    AStressTrainer(GRUaStress &model,
                   LossFunctionType &loss_function,
                   DataLoaderType1 &train_loader,
                   DataLoaderType2 &val_loader,
                   OptimizerType &optimizer,
                   LRSchedulertype &lr_scheduler,
                   torch::Device device,
                   torch::Dtype dtype = torch::kFloat64)
        : Trainer<GRUaStress, LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>(
              model,
              loss_function,
              train_loader,
              val_loader,
              optimizer,
              lr_scheduler,
              device,
              dtype)
    {
    }

protected:
    torch::Tensor calculate_loss(torch::Tensor data, torch::Tensor target) override
    {
        torch::Tensor t = data.select(-1, 0);
        torch::Tensor mask1 = (t >= 0);

        torch::Tensor output;
        std::tie(output, std::ignore) = this->_model->forward(data);
        auto strain_dim = data.size(-1) - 1;
        torch::Tensor pred_stress = output.slice(-1, 0, strain_dim);
        // torch::Tensor pred_jacobian = output.slice(-1, strain_dim);
        // torch::Tensor pred_jacobian0 = pred_jacobian.select(1, 0);
        torch::Tensor stress = target.slice(-1, 0, strain_dim);
        // torch::Tensor jacobian = target.slice(-1, strain_dim);
        // torch::Tensor jacobian0 = jacobian.select(1, 0);
        torch::Tensor loss = this->_loss_function->forward(pred_stress, stress);
        torch::Tensor mask = mask1.unsqueeze(-1).expand_as(stress);
        this->_loss_w = torch::sum(mask).item<int>();
        return loss.masked_select(mask).mean();
    }

    void lr_scheduler_step(double val_loss) override
    {
        this->_lr_scheduler.step(val_loss);
    }
};

template <typename LossFunctionType, typename DataLoaderType1, typename DataLoaderType2, typename OptimizerType, typename LRSchedulertype>
AStressTrainer<LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>
make_astress_trainer(GRUaStress &model,
                     LossFunctionType &loss_function,
                     DataLoaderType1 &train_loader,
                     DataLoaderType2 &val_loader,
                     OptimizerType &optimizer,
                     LRSchedulertype &lr_scheduler,
                     torch::Device device,
                     torch::Dtype dtype = torch::kFloat64)

{
    return AStressTrainer<LossFunctionType,
                          DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>(model,
                                                                                            loss_function,
                                                                                            train_loader,
                                                                                            val_loader,
                                                                                            optimizer,
                                                                                            lr_scheduler,
                                                                                            device,
                                                                                            dtype);
}
#pragma once

#include "Trainer.h"
#include "GRUStress.h"

template <typename LossFunctionType, typename DataLoaderType1, typename DataLoaderType2, typename OptimizerType, typename LRSchedulertype>
class StressTrainer : public Trainer<GRUStress, LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>
{
public:
    StressTrainer(GRUStress &model,
                  LossFunctionType &loss_function,
                  DataLoaderType1 &train_loader,
                  DataLoaderType2 &val_loader,
                  OptimizerType &optimizer,
                  LRSchedulertype &lr_scheduler,
                  torch::Device device,
                  torch::Dtype dtype = torch::kFloat64)
        : Trainer<GRUStress, LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>(
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
    virtual torch::Tensor calculate_loss(torch::Tensor data, torch::Tensor target) override
    {
        // {batch, seq}
        torch::Tensor t = data.select(-1, 0);
        torch::Tensor mask1 = (t >= 0);

        torch::Tensor output;
        std::tie(output, std::ignore) = this->_model->forward(data);
        torch::Tensor loss = this->_loss_function->forward(output, target);

        torch::Tensor mask = mask1.unsqueeze(-1).expand_as(target);

        this->_loss_w = torch::sum(mask).item<int>();

        return loss.masked_select(mask).mean();
    }

    virtual std::tuple<torch::Tensor, torch::Tensor> calculate_train_loss(torch::Tensor data, torch::Tensor target) override
    {
        auto a = this->calculate_loss(data, target);
        return std::make_tuple(a, a);
        // {batch, seq}
        torch::Tensor t = data.select(-1, 0);
        torch::Tensor mask1 = (t >= 0);

        torch::Tensor strain = data.slice(-1, 1).set_requires_grad(true);
        torch::Tensor data_g = torch::cat({t.unsqueeze(-1), strain}, -1);
        // torch::Tensor data_g = data.set_requires_grad(true);

        torch::Tensor output;
        std::tie(output, std::ignore) = this->_model->forward(data_g);
        torch::Tensor loss = this->_loss_function->forward(output, target);

        torch::Tensor mask = mask1.unsqueeze(-1).expand_as(target);

        // torch::Tensor grad_reg = torch::autograd::grad({output}, {strain}, {torch::ones_like(output)}, true)[0];

        torch::Tensor loss_main = loss.masked_select(mask).mean();
        // torch::Tensor loss_grad = grad_reg.pow(2).masked_select(mask).mean();

        this->_loss_w = torch::sum(mask).item<int>();

        return std::make_tuple(loss_main /*+ 0.0 * loss_grad*/, loss_main);
    }

    void lr_scheduler_step(double val_loss) override
    {
        this->_lr_scheduler.step(val_loss);
    }
};

template <typename LossFunctionType, typename DataLoaderType1, typename DataLoaderType2, typename OptimizerType, typename LRSchedulertype>
StressTrainer<LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>
make_stress_trainer(GRUStress &model,
                    LossFunctionType &loss_function,
                    DataLoaderType1 &train_loader,
                    DataLoaderType2 &val_loader,
                    OptimizerType &optimizer,
                    LRSchedulertype &lr_scheduler,
                    torch::Device device,
                    torch::Dtype dtype = torch::kFloat64)

{
    return StressTrainer<LossFunctionType,
                         DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>(model,
                                                                                           loss_function,
                                                                                           train_loader,
                                                                                           val_loader,
                                                                                           optimizer,
                                                                                           lr_scheduler,
                                                                                           device,
                                                                                           dtype);
}

#pragma once

#include <cstddef>
#include "Neuron.h"
#include "Layer.h"

class MultiLayerPerceptron {
public:
    // Constructors
    MultiLayerPerceptron(size_t nIn, std::vector<size_t> nOuts);

    // Parameters
    std::shared_ptr<std::vector<std::shared_ptr<Value>>> getParameters();

    // Functor
    Value* operator()(Value &input);
private:
    std::vector<std::shared_ptr<Layer>> mLayers;
};
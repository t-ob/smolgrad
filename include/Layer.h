#pragma once

#include <cstddef>
#include "Neuron.h"

class Layer {
public:
    // Constructors
    Layer(size_t nIn, size_t nOut);

    // Parameters
    std::shared_ptr<std::vector<std::shared_ptr<Value>>> getParameters();

    // Functor
    Value* operator()(Value &input);

private:
    size_t mNIn;
    size_t mNOut;
    std::vector<std::shared_ptr<Neuron>> mNeurons;
};

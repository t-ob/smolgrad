#pragma once

#include <cstddef>
#include "Value.h"

class Neuron {
public:
    // Constructors
    Neuron(size_t nIn);

    // Parameters
    std::shared_ptr<std::vector<std::shared_ptr<Value>>> getParameters();

    // Functor
    Value* operator()(Value &input);

private:
    size_t mNIn;
    std::shared_ptr<Value> mWeight;
    std::shared_ptr<Value> mBias;
};

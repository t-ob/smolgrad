//
// Created by tom on 06/07/23.
//
#include "Neuron.h"

Neuron::Neuron(size_t nIn) : mNIn(nIn), mWeight(std::make_shared<Value>(Value::rand(nIn, -1.0f, 1.0f))), mBias(std::make_shared<Value>(Value::rand(1, -1.0f, 1.0f))) {

}

Value* Neuron::operator()(Value &input) {
    auto c = mWeight->dot(input);
    auto d = *c + *mBias;
    return d->tanh();
}

std::shared_ptr<std::vector<std::shared_ptr<Value>>> Neuron::getParameters() {
    auto parameters = std::make_shared<std::vector<std::shared_ptr<Value>>>();
    parameters->push_back(mWeight);
    parameters->push_back(mBias);

    return parameters;
}

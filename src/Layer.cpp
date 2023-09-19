//
// Created by tom on 20/07/23.
//
#include "Layer.h"

Layer::Layer(size_t nIn, size_t nOut) : mNIn(nIn), mNOut(nOut) {
    for (size_t i = 0; i < nOut; i++) {
        mNeurons.push_back(std::make_shared<Neuron>(nIn));
    }
}

Value *Layer::operator()(Value &input) {
    std::vector<Value*> outputs;
    for (auto &neuron : mNeurons) {
        outputs.push_back((*neuron)(input));
    }
    return Value::concat(outputs);
}

std::shared_ptr<std::vector<std::shared_ptr<Value>>> Layer::getParameters() {
    auto result = std::make_shared<std::vector<std::shared_ptr<Value>>>();

    for (auto &neuron : mNeurons) {
        auto parameters = neuron->getParameters();
        for (auto &parameter : *parameters) {
            result->push_back(parameter);
        }
//        result->insert(result->end(), parameters->begin(), parameters->end());
    }

    return result;
}

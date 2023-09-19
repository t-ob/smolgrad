//
// Created by tom on 20/07/23.
//
#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(size_t nIn, std::vector<size_t> nOuts) {
    auto currIn = nIn;
    for (auto &nOut : nOuts) {
        mLayers.push_back(std::make_shared<Layer>(currIn, nOut));
        currIn = nOut;
    }
}

Value *MultiLayerPerceptron::operator()(Value &input) {
    auto currInput = new Value(input);
    for (auto &layer : mLayers) {
        currInput = (*layer)(*currInput);
    }
    return currInput;
}

std::shared_ptr<std::vector<std::shared_ptr<Value>>> MultiLayerPerceptron::getParameters() {
    auto result = std::make_shared<std::vector<std::shared_ptr<Value>>>(std::vector<std::shared_ptr<Value>>());

    for (auto &layer : mLayers) {
        auto parameters = layer->getParameters();
        result->insert(result->end(), parameters->begin(), parameters->end());
    }

    return result;
}

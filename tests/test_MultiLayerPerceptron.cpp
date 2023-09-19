//
// Created by tom on 06/07/23.
//
#include <gtest/gtest.h>
#include "MultiLayerPerceptron.h"
#include "Value.h"

TEST(TestMultiLayerPerceptronConstructor, TestMultiLayerPerceptronConstructorDoesNotRaise) {
    MultiLayerPerceptron mlp(10, {20, 20, 10});
}

TEST(TestMultiLayerPerceptronFunctor, TestMultiLayerPerceptronFunctorDoesNotRaise) {
    MultiLayerPerceptron mlp(10, {20, 20, 10});
    auto input = Value::constant(10, 1.0f);
    mlp(input);
}

TEST(FOO, BAR) {
    MultiLayerPerceptron mlp(2, {20, 20, 10, 1});

    auto inputs = std::vector<Value> {
            Value {0.5f, 0.1f},
            Value {0.7f, 1.0f},
            Value {0.1f, -0.2f},
            Value {-0.1f, 1.0f},
            Value {-0.5f, -0.1f},
            Value {-0.3f, 0.2f}
    };

    auto expected = std::vector<Value> {
            Value {1.0f},
            Value {0.0f},
            Value {1.0f},
            Value {0.0f},
            Value {1.0f},
            Value {1.0f}
    };

    std::vector<Value> observed;

    for (Value &input : inputs) {
        observed.push_back(*mlp(input));
    }

    // Calculate loss
    auto loss = Value::constant(1, 0.0f);
    for (size_t i = 0; i < inputs.size(); i++) {
        auto diff = observed[i] - expected[i];
        auto square = diff->pow(2.0f);
        loss = *(loss + *square);
    }

    // Calculate gradient
    loss.backward();

    // Update parameters
    auto parameters = mlp.getParameters();
    for (auto parameter : *parameters) {
        auto grad = parameter->getGrad();

        auto wow = parameter;
        auto lol = *grad * 0.01f;

        auto sos = *wow - *lol;

        *parameter = *sos;

        parameter->clearGrad();
    }

}

//
// Created by tom on 20/07/23.
//
#include <iostream>

#include "MultiLayerPerceptron.h"
#include "Neuron.h"
#include "Value.h"

// render a vector of floats
std::ostream &operator<<(std::ostream &os, const std::vector<float> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


// main function
int main(int argc, char **argv) {
    auto neuron = Neuron(2);

    // print neuron params
    std::cout << "Neuron params: " << std::endl;
    auto ps = neuron.getParameters();
    for (auto &p : *ps) {
        auto lll = p->getData();
        auto jjj = p->getGrad()->getData();
        std::cout << "Values: " << *lll << std::endl;
        std::cout << "Grad: " << *jjj << std::endl;
    }

    auto in_1 = Value {1.0, 2.0};

    auto out_1 = neuron(in_1);

    out_1->backward();
    std::cout << "Neuron params: " << std::endl;
    ps = neuron.getParameters();
    for (auto &p : *ps) {
        auto lll = p->getData();
        auto jjj = p->getGrad()->getData();
        std::cout << "Values: " << *lll << std::endl;
        std::cout << "Grad: " << *jjj << std::endl;
    }

    for (auto &p : *ps) {
        *p -= *(*p->getGrad() * 0.01);
        p->clearGrad();
    }
    for (auto &p : *ps) {
        auto lll = p->getData();
        auto jjj = p->getGrad()->getData();
        std::cout << "Values: " << *lll << std::endl;
        std::cout << "Grad: " << *jjj << std::endl;
    }

    std::cout << "neuron test done, starting layer test" << std::endl;

    auto layer = Layer(2, 3);

    // print layer params
    std::cout << "Layer params: " << std::endl;
    ps = layer.getParameters();
    for (auto &p : *ps) {
        auto lll = p->getData();
        auto jjj = p->getGrad()->getData();
        std::cout << "Values: " << *lll << std::endl;
        std::cout << "Grad: " << *jjj << std::endl;
    }

    auto in_2 = Value {1.0, 2.0};
    auto out_2 = layer(in_2);

    out_2->backward();

    std::cout << "Layer params: " << std::endl;
    ps = layer.getParameters();
    for (auto &p : *ps) {
        auto lll = p->getData();
        auto jjj = p->getGrad()->getData();
        std::cout << "Values: " << *lll << std::endl;
        std::cout << "Grad: " << *jjj << std::endl;
    }

    std::cout << "Layer test done, starting MLP test" << std::endl;

    MultiLayerPerceptron mlp(2, {20, 20, 10, 1});

    auto inputs = std::vector<Value *> {
            new Value {0.5f, 0.1f},
            new Value {0.7f, 1.0f},
            new Value {0.1f, -0.2f},
            new Value {-0.1f, 1.0f},
            new Value {-0.5f, -0.1f},
            new Value {-0.3f, 0.2f}
    };

    auto expected = std::vector<Value *> {
            new Value {1.0f},
            new Value {0.0f},
            new Value {1.0f},
            new Value {0.0f},
            new Value {1.0f},
            new Value {1.0f}
    };

    for (size_t step = 0; step < 1000; ++step) {

        std::vector<Value *> observed;

        for (auto input: inputs) {
            observed.push_back(mlp(*input));
        }

        auto o = Value::concat(observed);
        auto e = Value::concat(expected);

        auto diff = *o - *e;
        auto l = diff->dot(*diff);


        // Print loss
//        std::cout << "Loss: " << *l->getData() << std::endl;

        // Calculate gradient
        l->backward();

        // Update parameters
        auto parameters = mlp.getParameters();
        for (auto &parameter: *parameters) {
//            auto grad = parameter->getGrad();
//
//            auto wow = parameter;
//            auto lol = *grad * 0.01f;
//
//            auto sos = *wow - *lol;
//
//            *parameter = *sos;

            *parameter -= *(*parameter->getGrad() * 0.01);

            parameter->clearGrad();
        }
    }

    std::cout << "Hello, world!" << std::endl;
}
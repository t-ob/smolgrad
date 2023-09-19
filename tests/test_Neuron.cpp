//
// Created by tom on 06/07/23.
//
#include <gtest/gtest.h>
#include "Neuron.h"

TEST(TestConstructor, TestConstructorDoesNotRaise) {
    Neuron neuron(10);
}

TEST(TestFunctor, TestFunctorDoesNotRaise) {
    Neuron neuron(10);
    auto input = Value::constant(10, 1.0f);
    neuron(input);
}
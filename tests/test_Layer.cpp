//
// Created by tom on 06/07/23.
//
#include <gtest/gtest.h>
#include "Layer.h"

TEST(TestLayerConstructor, TestLayerConstructorDoesNotRaise) {
Layer layer(10, 20);
}

TEST(TestLayerFunctor, TestLayerFunctorDoesNotRaise) {
Layer layer(10, 20);
auto input = Value::constant(10, 1.0f);
layer(input);
}

TEST(TestLayerFunctor, TestLayerFunctorShape) {
    Layer layer(10, 20);
    auto input = Value::constant(10, 1.0f);
    auto output = layer(input);

    EXPECT_EQ(output->getData()->size(), 20);
}
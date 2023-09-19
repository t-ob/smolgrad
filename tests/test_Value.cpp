//
// Created by tom on 30/06/23.
//
#include <gtest/gtest.h>
#include "Value.h"

TEST(TestValue, TestSize) {
Value value(10);
EXPECT_EQ(10, value.getSize());

Value value2 {1.0f, 2.0f, 3.0f};
EXPECT_EQ(3, value2.getSize());

auto grad = value2.getGrad()->getData();
    EXPECT_EQ(0.0f, (*grad)[0]);
    EXPECT_EQ(0.0f, (*grad)[1]);
    EXPECT_EQ(0.0f, (*grad)[2]);
}

TEST(TestValue, TestConstant) {
Value value = Value::constant(10, 1.0f);
for (size_t i = 0; i < 10; ++i)
{
EXPECT_EQ(1.0f, value[i]);
}
}

TEST(TestValue, TestArithmetic) {
    auto a = new Value {2.0f};
    auto b = new Value {-3.0f};
    auto c = new Value {10.f};
    auto e = *a * *b; // e = -6
    auto d = *e + *c; // d = 4
    auto f = new Value {-2.0f};
    auto L = *d * *f; // L = -8

    auto data = *L->getData();

    ASSERT_EQ(data[0], -8.0f);
}

TEST(TestBackward, TestPlus) {
    auto a = new Value {2.0f};
    auto b = new Value {-3.0f};
    auto c = *a + *b;
    c->backward();

    auto grad = *a->getGrad()->getData();
    ASSERT_EQ(grad[0], 1.0f);

    grad = *b->getGrad()->getData();
    ASSERT_EQ(grad[0], 1.0f);
}

TEST(TestBackward, TestMinus) {
    auto a = new Value {2.0f};
    auto b = new Value {-3.0f};
    auto c = *a - *b;
    c->backward();

    auto grad = *a->getGrad()->getData();
    ASSERT_EQ(grad[0], 1.0f);

    grad = *b->getGrad()->getData();
    ASSERT_EQ(grad[0], -1.0f);
}

TEST(TestBackward, TestMul) {
    auto a = new Value {2.0f};
    auto b = new Value {-3.0f};
    auto c = *a * *b;
    c->backward();

    auto grad = *a->getGrad()->getData();
    ASSERT_EQ(grad[0], -3.0f);

    grad = *b->getGrad()->getData();
    ASSERT_EQ(grad[0], 2.0f);
}

TEST(TestBackward, TestDiv) {
    // Positive numerator, positive denominator
    auto a = new Value {2.0f};
    auto b = new Value {3.0f};
    auto c = *a / *b;
    c->backward();

    auto grad = *a->getGrad()->getData();
    EXPECT_NEAR(grad[0], 1.0f / 3.0f, 1e-5);

    grad = *b->getGrad()->getData();
    EXPECT_NEAR(grad[0], -2.0f / 9.0f, 1e-5);

    // Negative numerator, positive denominator
    a = new Value {-2.0f};
    b = new Value {3.0f};
    c = *a / *b;
    c->backward();

    grad = *a->getGrad()->getData();
    EXPECT_NEAR(grad[0], 1.0f / 3.0f, 1e-5);

    grad = *b->getGrad()->getData();
    EXPECT_NEAR(grad[0], 2.0f / 9.0f, 1e-5);

    // Positive numerator, negative denominator
    a = new Value {2.0f};
    b = new Value {-3.0f};
    c = *a / *b;
    c->backward();

    grad = *a->getGrad()->getData();
    EXPECT_NEAR(grad[0], -1.0f / 3.0f, 1e-5);

    grad = *b->getGrad()->getData();
    EXPECT_NEAR(grad[0], -2.0f / 9.0f, 1e-5);
}

TEST(TestDot, TestValue) {
    auto a = new Value {1.0f, 1.0f};
    auto b = new Value {2.0f, 3.0f};

    auto c = a->dot(*b);
    c->backward();

    auto grad_a = *a->getGrad()->getData();
    auto grad_b = *b->getGrad()->getData();

    ASSERT_EQ(grad_a[0], 2.0f);
    ASSERT_EQ(grad_a[1], 3.0f);
    ASSERT_EQ(grad_b[0], 1.0f);
    ASSERT_EQ(grad_b[1], 1.0f);
}

TEST(TestDot, TestGradients) {
    auto a = new Value {1.0f, 1.0f};
    auto b = new Value {2.0f, 3.0f};

    auto c = a->dot(*b);
    auto c_data = *c->getData();

    ASSERT_EQ(c_data[0], 5.0f);
}
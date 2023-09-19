#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include "Value.h"

// Factory methods
Value Value::constant(size_t size, float value) {
    Value constant(size);

    for (size_t i = 0; i < size; ++i) {
        constant.mData[i] = value;
    }

    return constant;
}

Value Value::rand(size_t size, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    Value rand(size);

    for (size_t i = 0; i < size; ++i) {
        rand.mData[i] = dis(gen);
    }

    return rand;
}

Value Value::randn(size_t size, float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);

    Value randn(size);

    for (size_t i = 0; i < size; ++i) {
        randn.mData[i] = dis(gen);
    }

    return randn;
}

// Constructor
Value::Value(size_t size)
        : mData(new float[size]()), mGrad(new float[size]()), mSize(size)  // Initialize `value` with `initialValue`
{
}

Value::Value(std::initializer_list<float> values)
        : mData(new float[values.size()]), mGrad(new float[values.size()]()), mSize(values.size()) {
    std::copy(values.begin(), values.end(), &mData[0]);
}

Value::Value(size_t size, std::initializer_list<Value *> refs)
        : mData(new float[size]()), mGrad(new float[size]()), mSize(size), mReferences(refs) {

}

Value::Value(size_t size, std::vector<Value *> &refs)
        : mData(new float[size]()), mGrad(new float[size]()), mSize(size), mReferences(refs) {

}

Value::Value(size_t size, float *values)
    : mData(new float[size]()), mGrad(new float[size]()), mSize(size) {
    std::copy(values, values + size, &mData[0]);
}


// Copy constructor
Value::Value(const Value &other)
        : mData(new float[other.mSize]), mGrad(new float[other.mSize]()), mSize(other.mSize) {
    std::copy(&other.mData[0], &other.mData[0] + other.mSize, &mData[0]);
    std::copy(&other.mGrad[0], &other.mGrad[0] + other.mSize, &mGrad[0]);
}

// Copy assignment operator
Value &Value::operator=(Value other) {
    std::swap(mData, other.mData);
    std::swap(mGrad, other.mGrad);
    std::swap(mSize, other.mSize);
    return *this;
}

// Subscript operator
float &Value::operator[](std::size_t index) {
    if (index >= mSize) {
        throw std::out_of_range("index out of range");
    }
    return mData[index];
}

const float &Value::operator[](std::size_t index) const {
    if (index >= mSize) {
        throw std::out_of_range("index out of range");
    }
    return mData[index];
}

// Arithmetic operators
Value *Value::operator+(Value &other) {
    if (mSize != other.mSize) {
        throw std::logic_error("size mismatch");
    }

    auto result = new Value(mSize, {this, &other});
    result->setBackward([result, this, &other]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i];
            other.mGrad[i] += result->mGrad[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] + other.mData[i];
    }

    return result;
}

Value *Value::operator-(Value &other) {
    return *this + *(-(other));
}

Value *Value::operator*(Value &other) {
    if (mSize != other.mSize) {
        throw std::logic_error("size mismatch");
    }

    auto result = new Value(mSize, {this, &other});
    result->setBackward([result, this, &other]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i] * other.mData[i];
            other.mGrad[i] += result->mGrad[i] * mData[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] * other.mData[i];
    }

    return result;
}

Value *Value::operator/(Value &other) {
    return *this * *other.pow(-1.0f);
}

Value *Value::operator-() {
    return *this * -1.0f;
}

Value *Value::operator+(float other) {
    auto result = new Value(mSize, {this});
    result->setBackward([result, this]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] + other;
    }

    return result;
}

Value *Value::operator-(float other) {
    return *this + (-other);
}

Value *Value::operator*(float other) {
    auto result = new Value(mSize, {this});
    result->setBackward([result, this, other]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i] * other;
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] * other;
    }

    return result;
}

Value *Value::operator/(float other) {
    return *this * (1.0f / other);
}

Value *Value::pow(float exponent) {
    auto result = new Value(mSize, {this});
    result->setBackward([result, this, exponent]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i] * exponent * std::pow(mData[i], exponent - 1.0f);
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = std::pow(mData[i], exponent);
    }

    return result;
}

Value *Value::exp() {
    auto result = new Value(mSize, {this});
    result->setBackward([result, this]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i] * result->mData[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = std::exp(mData[i]);
    }

    return result;
}

Value *Value::tanh() {
    auto result = new Value(mSize, {this});
    result->setBackward([result, this]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i] * (1.0f - result->mData[i] * result->mData[i]);
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = std::tanh(mData[i]);
    }

    return result;
}

// Member function
size_t Value::getSize() const {
    return mSize;
}

std::unique_ptr<std::vector<float>> Value::getData() {
    auto result = std::make_unique<std::vector<float>>();
    for (size_t i = 0; i < mSize; ++i) {
        result->push_back(mData[i]);
    }

    return result;
}

Value *Value::getGrad() {
    return new Value(mSize, mGrad.get());
}


// Output stream
std::ostream &operator<<(std::ostream &os, const Value &obj) {
    // write obj to stream
    os << "Value: " << obj.getSize();
    return os;
}

void Value::setBackward(std::function<void()> backward) {
    mBackward = backward;
}

void Value::backward() {
    struct Helper {
        static void sort(Value *value, std::unordered_set<Value *> &visited, std::vector<Value *> &visiting, std::vector<Value *> &result) {
            if (visited.find(value) != visited.end()) {
                return;
            }
            visited.insert(value);
            visiting.push_back(value);
            for (auto &ref: value->mReferences) {
                sort(ref, visited, visiting, result);
            }
            result.push_back(value);
            visiting.pop_back();
        }
    };

    for (size_t i = 0; i < mSize; ++i) {
        mGrad[i] = 1.0f;
    }

    std::unordered_set<Value *> visited;
    std::vector<Value *> visiting;
    std::vector<Value *> sorted;
    Helper::sort(this, visited, visiting, sorted);

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        if ((*it)->mBackward) {
            (*it)->mBackward();
        }
    }
}

Value *Value::sum() {
    auto result = new Value(1, {this});
    result->setBackward([result, this]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[0];
        }
    });

    float sum = 0.0f;
    for (size_t i = 0; i < mSize; ++i) {
        sum += mData[i];
    }
    result->mData[0] = sum;

    return result;
}

Value *Value::dot(Value &other) {
    auto result = *this * other;
    return result->sum();
}

float Value::at(size_t index) const {
    return mData[index];
}

Value *Value::concat(std::initializer_list<Value *> values) {
    size_t size = 0;
    for (auto &value: values) {
        size += value->getSize();
    }

    auto result = new Value(size, values);
    result->setBackward([result, values]() {
        size_t offset = 0;
        for (auto &value: values) {
            for (size_t i = 0; i < value->getSize(); ++i) {
                value->mGrad[i] += result->mGrad[offset + i];
            }
            offset += value->getSize();
        }
    });

    size_t offset = 0;
    for (auto &value: values) {
        for (size_t i = 0; i < value->getSize(); ++i) {
            result->mData[offset + i] = value->mData[i];
        }
        offset += value->getSize();
    }

    return result;
}

Value *Value::concat(std::vector<Value *> &values) {
    size_t size = 0;
    for (auto &value: values) {
        size += value->getSize();
    }

    auto result = new Value(size, values);
    result->setBackward([result, values]() {
        size_t offset = 0;
        for (auto &value: values) {
            for (size_t i = 0; i < value->getSize(); ++i) {
                value->mGrad[i] += result->mGrad[offset + i];
            }
            offset += value->getSize();
        }
    });

    size_t offset = 0;
    for (auto &value: values) {
        for (size_t i = 0; i < value->getSize(); ++i) {
            result->mData[offset + i] = value->mData[i];
        }
        offset += value->getSize();
    }

    return result;
}

void Value::clearGrad() {
    for (size_t i = 0; i < mSize; ++i) {
        mGrad[i] = 0.0f;
    }
}

void Value::operator+=(Value &other) {
    for (size_t i = 0; i < mSize; ++i) {
        mData[i] += other.mData[i];
    }
}

void Value::operator-=(Value &other) {
    for (size_t i = 0; i < mSize; ++i) {
        mData[i] -= other.mData[i];
    }
}

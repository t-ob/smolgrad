#include <cstddef>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "Value.h"

// Factory methods
Value Value::constant(size_t size, float value) {
    Value constant(size);

    for (size_t i = 0; i < size; ++i) {
        constant.mData[i] = value;
    }

    return constant;
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

Value::Value(size_t size, std::initializer_list<Value *> refs, Operation op)
        : mData(new float[size]()), mGrad(new float[size]()), mSize(size), mReferences(refs), mOperation(op) {

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

    auto result = new Value(mSize, {this, &other}, Operation::ADD);
    result->setBackward([result, this, &other]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i];
        }
        for (size_t i = 0; i < mSize; ++i) {
            other.mGrad[i] += result->mGrad[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] + other.mData[i];
    }

    return result;
}

Value *Value::operator-(Value &other) {
    if (mSize != other.mSize) {
        throw std::logic_error("size mismatch");
    }

    auto result = new Value(mSize, {this, &other}, Operation::SUB);
    result->setBackward([result, this, &other]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i];
        }
        for (size_t i = 0; i < mSize; ++i) {
            other.mGrad[i] -= result->mGrad[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] - other.mData[i];
    }

    return result;
}

Value *Value::operator*(Value &other) {
    if (mSize != other.mSize) {
        throw std::logic_error("size mismatch");
    }

    auto result = new Value(mSize, {this, &other}, Operation::MUL);
    result->setBackward([result, this, &other]() {
        for (size_t i = 0; i < mSize; ++i) {
            mGrad[i] += result->mGrad[i] * other.mData[i];
        }
        for (size_t i = 0; i < mSize; ++i) {
            other.mGrad[i] += result->mGrad[i] * mData[i];
        }
    });

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] * other.mData[i];
    }

    return result;
}

Value *Value::operator/(Value &other) {
    if (mSize != other.mSize) {
        throw std::logic_error("size mismatch");
    }

    auto result = new Value(mSize, {this, &other}, Operation::DIV);

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] / other.mData[i];
    }

    return result;
}

Value *Value::operator-() {
    auto result = new Value(mSize, {this}, Operation::NEG);

    for (size_t i = 0; i < mSize; ++i) {
        result[i] = -mData[i];
    }

    return result;
}

Value *Value::operator+(float other) {
    auto result = new Value(mSize, {this}, Operation::ADD);

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] + other;
    }

    return result;
}

Value *Value::operator-(float other) {
    auto result = new Value(mSize, {this}, Operation::SUB);

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] - other;
    }

    return result;
}

Value *Value::operator*(float other) {
    auto result = new Value(mSize, {this}, Operation::MUL);

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] * other;
    }

    return result;
}

Value *Value::operator/(float other) {
    auto result = new Value(mSize, {this}, Operation::DIV);

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = mData[i] / other;
    }

    return result;
}

Value *Value::pow(float exponent) {
    auto result = new Value(mSize, {this}, Operation::POW);

    for (size_t i = 0; i < mSize; ++i) {
        result->mData[i] = std::pow(mData[i], exponent);
    }

    return result;
}

// Member function
size_t Value::getSize() const {
    return mSize;
}

std::unique_ptr<float[]> &Value::getData() {
    return mData;
}

std::unique_ptr<float[]> &Value::getGrad() {
    return mGrad;
}

// Graphviz DOT output
//std::string Value::toDot() const {
//    // Get all nodes in graph
//    std::vector<std::shared_ptr<const Value>> nodes;
//    std::vector<std::shared_ptr<const Value>> queue;
//    queue.push_back(this->shared_from_this());
//    while (!queue.empty()) {
//        auto node = queue.back();
//        queue.pop_back();
//
//        if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
//            nodes.push_back(node);
//            for (auto &ref : node->references) {
//                queue.push_back(ref);
//            }
//        }
//    }
//
//    // Get all edges in graph
//    std::vector<std::pair<std::shared_ptr<const Value>, std::shared_ptr<const Value>>> edges;
//    for (auto &node : nodes) {
//        for (auto &ref : node->references) {
//            edges.emplace_back(node, ref);
//        }
//    }
//
//    // Generate DOT
//    std::string dot = "digraph G {\n";
//    for (auto &node : nodes) {
//        dot += "    " + std::to_string(node.get()) + " [label=\"" + node->operationToString() + "\"];\n";
//        if (node->mOperation) {
//            dot += "    " + std::to_string(node.get()) + " [shape=box];\n";
//        }
//    }
//    for (auto &edge : edges) {
//        dot += "    " + std::to_string(edge.first.get()) + " -> " + std::to_string(edge.second.get()) + ";\n";
//    }
//    dot += "}\n";
//
//    return dot;
//}

// Output stream
std::ostream &operator<<(std::ostream &os, const Value &obj) {
    // write obj to stream
    os << "MyClass: " << obj.getSize();
    return os;
}

void Value::setBackward(std::function<void()> backward) {
    mBackward = backward;
}

void Value::backward() {
    struct Helper {
        static void sort(Value *value, std::vector<Value *> &visiting, std::vector<Value *> &visited) {
            visiting.push_back(value);
            for (auto &ref: value->mReferences) {
                sort(ref, visiting, visited);
            }
            visited.push_back(value);
            visiting.pop_back();
        }
    };

    for (size_t i = 0; i < mSize; ++i) {
        mGrad[i] = 1.0f;
    }

    std::vector<Value *> visiting;
    std::vector<Value *> visited;
    Helper::sort(this, visiting, visited);

    for (auto it = visited.rbegin(); it != visited.rend(); ++it) {
        if ((*it)->mBackward) {
            (*it)->mBackward();
        }
    }
}

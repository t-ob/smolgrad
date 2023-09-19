#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

class Value
{
public:
    // Factory methods
    static Value constant(size_t size, float value);
    static Value rand(size_t size, float min, float max);
    static Value randn(size_t size, float mean, float stddev);

    static Value* concat(std::initializer_list<Value*> values);
    static Value* concat(std::vector<Value*>& values);

    // Constructor
    Value(size_t size);
    Value(std::initializer_list<float> values);
    Value(size_t size, std::initializer_list<Value*> refs);
    Value(size_t size, std::vector<Value*> &refs);
    Value(size_t size, float *values);


    // Copy constructor
    Value(const Value& other);

    // Copy assignment operator
    Value& operator=(Value other);

    // Subscript operator
    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    // Arithmetic operators
    Value* operator+(Value& other);
    Value* operator-(Value& other);
    Value* operator*(Value& other);
    Value* operator/(Value& other);
    Value* operator-();

    void operator+=(Value& other);
    void operator-=(Value& other);

    Value* operator+(float other);
    Value* operator-(float other);
    Value* operator*(float other);
    Value* operator/(float other);

    Value* pow(float exponent);
    Value* exp();
    Value* tanh();

    // Member function
    size_t getSize() const;
    std::unique_ptr<std::vector<float>> getData();
    Value* getGrad();
    float at(size_t index) const;

    // Set backward function
    void setBackward(std::function<void()> backward);

    // Backward pass
    void backward();

    // Clear gradient
    void clearGrad();

    // Sum
    Value* sum();

    // Dot product
    Value *dot(Value &other);

    // << operator
    friend std::ostream& operator<<(std::ostream& os, const Value& value);

private:
    // Member variable
    size_t mSize;
    std::unique_ptr<float[]> mGrad;
    std::unique_ptr<float[]> mData;
    std::vector<Value*> mReferences;
    std::function<void()> mBackward;
};

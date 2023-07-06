#pragma once

#include <memory>
#include <optional>
#include <vector>

class Value : public std::enable_shared_from_this<Value>
{
public:
    // Factory methods
    static Value constant(size_t size, float value);
    static Value rand(size_t size, float min, float max);
    static Value randn(size_t size, float mean, float stddev);

    // Constructor
    Value(size_t size);
    Value(std::initializer_list<float> values);
    Value(size_t size, std::initializer_list<Value*> refs);


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

    Value* operator+(float other);
    Value* operator-(float other);
    Value* operator*(float other);
    Value* operator/(float other);

    Value* pow(float exponent);
    Value* exp();
    Value* tanh();

    // Member function
    size_t getSize() const;
    std::unique_ptr<float[]>& getData();
    std::unique_ptr<float[]>& getGrad();

    // Set backward function
    void setBackward(std::function<void()> backward);

    // Backward pass
    void backward();

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

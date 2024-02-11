/* 
A simple automatic differentiation (autograd) engine.
A reimplementation of Micrograd (https://github.com/karpathy/micrograd/tree/master/micrograd)
But will be expanded to Tensors.

c++ seems kind of bad for this structure of things... 
Maybe I won't use OO to implement the autograd engine, or switch to another language.
Or use some other less clunky way to represent the graphs, and build an OO interface for the graphs.
*/

#include <iostream>
#include <sstream>
#include <ostream>
#include <vector>
#include <unordered_set>
using namespace std;

/*
Value: a wrapper around a number. Each Value object has a '.grad' method
and a '.backward()' method. Calling .backward() on some expression evaluates
the derivative of each Value with respect to the expression and stores it in .grad.
*/
class Value {
        double data;
        double grad;
        function<double(void)> _backward;
         // any value will have at most 2 children, since we support at most binary operations
        shared_ptr<Value> child1, child2;
        bool has_children;

    public:
        Value(double data) : data(data), grad(0), has_children(false) {}
 
        Value(double data, Value& child1, Value& child2) {
            this->data = data;
            this->grad = 0;
            this->child1 = make_shared<Value>(child1);
            this->child2 = make_shared<Value>(child2);
            this->has_children = true;
        }

        Value operator+(Value& other) {
            Value out = Value(this->data + other.data, *this, other); 

            out._backward = []() -> double {
                return 0;
            };

            return out;
        }

        string to_string() {
            stringstream s; 
            s << "Value(data=" << this->data << ", grad=" << this->grad;
            if (this->has_children) {
                s << ", children=\n";
                s << "\t" << child1->to_string() << "\n";
                s << "\t" << child2->to_string() << "\n";
            }
            s << ")";
            return s.str();
        }
};

int main()
{
    Value a(4);
    Value b(20);

    Value c = a + b;

    cout << c.to_string() << endl;
}
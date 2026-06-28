#include"lattice.h"
#include"Model.h"



template<int Dim>
class MonteCarlo {
public:
    virtual void Metropolis();
    virtual void Geometry();
    virtual void PT();

    Model<Dim>* model = nullptr;
};

template<int Dim>
class MonteCarlo_SAW : MonteCarlo<Dim> {
public:
    MonteCarlo_SAW(Model<Dim>* model_) {
        this->model = model_;
    }
};

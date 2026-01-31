#include"lattice.h"
#include"Model.h"




class MonteCarlo {
public:
    virtual void Metropolis();
    virtual void Geometry();
    virtual void PT();

    Model* model = nullptr;
};


class MonteCarlo_SAW : MonteCarlo {
public:
    MonteCarlo_SAW(Model* model_) {
        model = model_;
    }
};

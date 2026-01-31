#include"Model.h"


//used to increase length of SAWs for lattice side
#ifndef OUT_Length
#define OUT_Length 4
#endif
//used to define lattice nodes without spins (SAW does not go over this node)
#ifndef NO_SAW_NODE
#define NO_SAW_NODE -1
#endif
//used to define lattice nodes without XY spins
#ifndef NO_XY_SPIN
#define NO_XY_SPIN -5
#endif



Model::Model(int L) : L(L) {
    #ifdef REGIME_2D
    lattice = new Lattice_2D(2 * L + OUT_Length);
#else
    lattice = new Lattice_3D(0.75*L+OUT_Length);
#endif
};

void Model::HostDataInit() {
    hostdata.map_of_contacts_int = lattice->map_of_contacts_int;
    hostdata.inverse_steps       = lattice->inverse_steps;

    
}

void Model::DeviceDataInit() {
    upload_all(hostdata, devicedata);
}
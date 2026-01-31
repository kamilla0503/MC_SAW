#include"common.h"
#include"lattice.h"


Lattice_2D::Lattice_2D(int max_seq_size) : Lattice(max_seq_size) {
    number_of_nodes = lattice_side*lattice_side;
    create_lattice();
}

Lattice_3D::Lattice_3D(int max_seq_size) : Lattice(max_seq_size) {
    number_of_nodes = lattice_side*lattice_side*lattice_side;
    create_lattice();
}

void Lattice_2D::create_lattice() {
    int x, y;
    div_t n;
    map_of_contacts_int.resize(lattice_side*lattice_side*ndim2());
    for (int i =0; i<number_of_nodes ; i++){
        map_of_contacts_int[ndim2()*i] = i+1;
        map_of_contacts_int[ndim2()*i+1] = i-1;
        map_of_contacts_int[ndim2()*i+2] = i+lattice_side;
        map_of_contacts_int[ndim2()*i+3] = i-lattice_side;
        n=div(i, lattice_side);
        x=n.rem;
        y=n.quot;
        for (int j =0; j<ndim2(); j++){
            if(x==0){
                map_of_contacts_int[ndim2()*i+1] = i+lattice_side-1;
            }
            if(x==(lattice_side-1)){
                map_of_contacts_int[ndim2()*i] = i-(lattice_side-1);
            }
            if(y==0){
                map_of_contacts_int[ndim2()*i+3] = lattice_side*(lattice_side-1)+x;
            }
            if(y==(lattice_side-1)){
                map_of_contacts_int[ndim2()*i+2] = x;
            }
        }
    }
    inverse_steps.resize(ndim2());
    inverse_steps = {1,0,3,2};
}

void Lattice_3D::create_lattice() {

}






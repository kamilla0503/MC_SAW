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
    map_of_contacts_int = Kokkos::View<int*, Kokkos::HostSpace>("map_of_contacts_int",
        lattice_side*lattice_side*ndim2());
    inverse_steps = Kokkos::View<int*, Kokkos::HostSpace>("inverse_steps", ndim2() );
    for (int i =0; i<number_of_nodes ; i++){
        map_of_contacts_int(ndim2()*i) = i+1;
        map_of_contacts_int(ndim2()*i+1) = i-1;
        map_of_contacts_int(ndim2()*i+2) = i+lattice_side;
        map_of_contacts_int(ndim2()*i+3) = i-lattice_side;
        n=div(i, lattice_side);
        x=n.rem;
        y=n.quot;
        for (int j =0; j<ndim2(); j++){
            if(x==0){
                map_of_contacts_int(ndim2()*i+1) = i+lattice_side-1;
            }
            if(x==(lattice_side-1)){
                map_of_contacts_int(ndim2()*i) = i-(lattice_side-1);
            }
            if(y==0){
                map_of_contacts_int(ndim2()*i+3) = lattice_side*(lattice_side-1)+x;
            }
            if(y==(lattice_side-1)){
                map_of_contacts_int(ndim2()*i+2) = x;
            }
        }
    }
    inverse_steps(0) = 1;
    inverse_steps(1) = 0;
    inverse_steps(2) = 3;
    inverse_steps(3) = 2;
}

void Lattice_3D::create_lattice() {
    int x, y,z;
    div_t n;
    int l;


    map_of_contacts_int = Kokkos::View<int*, Kokkos::HostSpace>("map_of_contacts_int",
        lattice_side*lattice_side*lattice_side*ndim2());
    inverse_steps = Kokkos::View<int*, Kokkos::HostSpace>("inverse_steps", ndim2() );


    for (int i =0; i<number_of_nodes ; i++){
        map_of_contacts_int(ndim2() * i) = i + 1;
        map_of_contacts_int(ndim2() * i + 1) = i - 1;
        map_of_contacts_int(ndim2() * i + 2) = i + lattice_side;
        map_of_contacts_int(ndim2() * i + 3) = i - lattice_side;
        map_of_contacts_int(ndim2() * i + 4) = i + lattice_side * lattice_side;
        map_of_contacts_int(ndim2() * i + 5) = i - lattice_side * lattice_side;

        l = lattice_side * lattice_side;
        n = div(i, l);
        z = n.quot;
        n = div( n.rem, lattice_side);
        x = n.rem;
        y = n.quot;

        for (int j = 0; j < ndim2(); j++) {
            if (x == 0) {
                map_of_contacts_int(ndim2() * i + 1) = i + lattice_side - 1;
            }
            if (x == (lattice_side - 1)) {
                map_of_contacts_int(ndim2() * i) = i - (lattice_side - 1);
            }
            if (y == 0) {
                map_of_contacts_int(ndim2() * i + 3) =  lattice_side * (lattice_side - 1) + i;
            }
            if (y == (lattice_side - 1)) {
                map_of_contacts_int(ndim2() * i + 2) = i - lattice_side * (lattice_side - 1) ;
            }
            if (z == 0) {
                map_of_contacts_int(ndim2() * i + 5) = i + lattice_side * lattice_side * (lattice_side - 1);
            }
            if (z == lattice_side - 1) {
                map_of_contacts_int(ndim2() * i + 4) = i - lattice_side * lattice_side * (lattice_side - 1);
            }
        }
    }
    inverse_steps(0) = 1;
    inverse_steps(1) = 0;
    inverse_steps(2) = 3;
    inverse_steps(3) = 2;
    inverse_steps(4) = 5;
    inverse_steps(5) = 4;
}

float Lattice_2D::radius(const int& start, const int& end) {
    int start_x = start % lattice_side;
    int start_y = start / lattice_side;
    int end_x = end % lattice_side;
    int end_y = end / lattice_side;

    //torus distance;
    float xdiff = abs(end_x - start_x);
    if (xdiff > (lattice_side/2))
        xdiff = lattice_side - xdiff;

    float ydiff = abs(end_y - start_y);
    if (ydiff > (lattice_side / 2))
        ydiff = lattice_side - ydiff;

    float r = xdiff *xdiff  + ydiff*ydiff ;

    return r;
}

float Lattice_3D::radius(const int& start, const int& end) {
    int start_x = start % lattice_side;
    int start_y = (start % (lattice_side * lattice_side)) /lattice_side;
    int start_z = start / (lattice_side * lattice_side);
    int end_x = end % lattice_side;
    int end_y = (end % (lattice_side * lattice_side)) /lattice_side;
    int end_z = end / (lattice_side * lattice_side);

    //torus distance;
    float xdiff = abs(end_x - start_x);
    if (xdiff > (lattice_side/2))
        xdiff = lattice_side - xdiff;

    float ydiff = abs(end_y - start_y);
    if (ydiff > (lattice_side / 2))
        ydiff = lattice_side - ydiff;

    float zdiff = abs(end_z - start_z);
    if (zdiff > (lattice_side / 2))
        zdiff = lattice_side - zdiff;

    float r = xdiff *xdiff  + ydiff*ydiff + zdiff*zdiff;

    return r;
}

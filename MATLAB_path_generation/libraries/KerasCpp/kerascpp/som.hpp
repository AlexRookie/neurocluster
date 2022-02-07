// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef SOM_H
#define SOM_H

// SOM layer

template <int Nin, int Nout>
void som(double in[Nin], const double prototypes[Nin][Nout], double out[Nout]) {
    for (int i=0; i<Nout; i++) {
        for (int j=0; j<Nin; j++) {
            out[i] += std::pow(in[j] - prototypes[j][i], 2);
        }
    }
}

#endif // SOM_H
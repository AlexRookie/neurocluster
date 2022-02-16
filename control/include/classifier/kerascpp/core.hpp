// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef CORE_H
#define CORE_H

// Keras core layers

template <int Nin, int Nout>
void dense(double in[Nin], const double w[Nout][Nin], const double b[Nout], double out[Nout]) {
    for (int j=0; j<Nout; j++) {
        out[j] = 0;
        for (int i=0; i<Nin; i++) {
            out[j] += in[i]*w[j][i];
        }
        out[j] += b[j];
    }
}

#endif // CORE_H
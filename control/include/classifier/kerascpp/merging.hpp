// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef MERGING_H
#define MERGING_H

// Keras merging layers

template<int NinA, int NinB>
void concatenate1d(double in_a[NinA], double in_b[NinB], double out[NinA+NinB]) {
    for (int i=0; i<NinA; i++) {
        out[i] = in_a[i];
    }
    for (int i=0; i<NinB; i++) {
        out[i+NinA] = in_b[i];
    }
}

#endif // MERGING_H
// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef ACTIVATION_H
#define ACTIVATION_H

// TODO aggiungi attivazioni da qui
// https://github.com/moof2k/kerasify/blob/master/keras_model.cc

// Keras activation layers

template <int Nin>
void relu(double in[Nin], double out[Nin]) {
    for(int i=0; i<Nin ; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

template <int Nin>
void leakyrelu(double in[Nin], double out[Nin], const double alpha) {
    for(int i=0; i<Nin ; i++) {
        out[i] = in[i] > 0 ? in[i] : alpha * in[i];
    }
}

template <int Nin>
void sigmoid(double in[Nin], double out[Nin]) {
    for(int i=0; i<Nin ; i++) {
        out[i] = 1 / (1 + std::exp(-in[i]) );
    }
}

template <int Nin>
void softmax(double in[Nin], double out[Nin]) {
    double sum = 0.0;
    for(int i=0; i<Nin ; i++) {
        sum += std::exp(in[i]);
    }
    for(int i=0; i<Nin ; i++) {
        out[i] = std::exp(in[i]) / sum;
    }
}

template <int Nin>
void tanh(double in[Nin], double out[Nin]) {
    for(int i=0; i<Nin ; i++) {
        out[i] = (std::exp(in[i]) - std::exp(-in[i])) / (std::exp(in[i]) + std::exp(-in[i])); //std::tanh(in[i]);
    }
}

template <int Nin>
void softplus(double in[Nin], double out[Nin]) {
    for(int i=0; i<Nin ; i++) {
        out[i] = std::log(std::exp(in[i]) + 1);
    }
}

template <int Nin>
void softsign(double in[Nin], double out[Nin]) {
    for(int i=0; i<Nin ; i++) {
        out[i] = in[i] / (std::abs(in[i]) + 1);
    }
}

template <int Nin>
void elu(double in[Nin], double out[Nin], const double alpha) {
    for(int i=0; i<Nin ; i++) {
        if (in[i] > 0) {
            out[i] = in[i];
        } else if (in[i] <= 0) {
            out[i] = alpha * (std::exp(in[i]) - 1);
        }
    }
}

template <int Nin>
void selu(double in[Nin], double out[Nin], const double alpha=1.67326324, const double scale=1.05070098) {
    for(int i=0; i<Nin ; i++) {
        if (in[i] > 0) {
            out[i] = scale * in[i];
        } else if (in[i] <= 0) {
            out[i] = scale * alpha * (std::exp(in[i]) - 1);
        }
    }
}

#endif // ACTIVATION_H
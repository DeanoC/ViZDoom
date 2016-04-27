//
// Created by deano on 27/04/16.
//

#ifndef VIZDOOM_BIASEDFULLYCONNECTEDNNLAYER_H
#define VIZDOOM_BIASEDFULLYCONNECTEDNNLAYER_H


class BiasedFullyConnectedNNLayer {
public:
    BiasedFullyConnectedNNLayer( int _inputSize, int _outputSize, int _batchSize );

    ~BiasedFullyConnectedNNLayer();

protected:
    const int inputSize;
    const int outputSize;
    int batchSize;

    std::vector <half_float> weights;
    std::vector <half_float> bias;

    cudnnTensorDescriptor_t tensorDescriptor;
    cudnnTensorDescriptor_t biasTensorDescriptor;

};


#endif //VIZDOOM_BIASEDFULLYCONNECTEDNNLAYER_H

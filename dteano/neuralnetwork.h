//
// Created by deano on 28/04/16.
//

#ifndef VIZDOOM_NEURALNETWORK_H
#define VIZDOOM_NEURALNETWORK_H

#include <vector>
#include <forward_list>
#include "neurallayer.h"
#include "cudacontext.h"

/*
 * Holds a neural network, currently restrict to a single item per layer
 */
class NeuralNetwork {
public:
    typedef std::shared_ptr< NeuralNetwork > ptr;

    NeuralNetwork( CudaContext::ptr _context, const TinyIndex _depth );

//    void presentTrainingFeature( float const * const input, float const * const float result );
//
//    void presentValidationFeature( float const * const input, float const * const expected_result );

    // input and output can be host or device memory
    void presentInput( half_or_float const *const input, half_or_float *output );

    using LayerContainer = std::vector< NeuralLayer::ptr >;

    size_t getInputCount() const { return (*layers.begin())->getInputCount(); }

    size_t getOutputCount() const { return layers.back()->getOutputCount(); }

    size_t getWeightCount() const;

    size_t getBiasWeightCount() const;

    LayerContainer::const_iterator cbegin() const { return layers.cbegin(); }

    LayerContainer::const_iterator cend() const { return layers.cend(); }

    LayerContainer::iterator begin() {
        dirty = true;
        return layers.begin();
    }

    LayerContainer::iterator end() { return layers.end(); }

    const NeuralLayer::ptr &operator[]( const TinyIndex _layer ) const {
        return layers.at(_layer);
    }

    NeuralLayer::ptr &operator[]( const TinyIndex _layer ) {
        dirty = true;
        return layers.at(_layer);
    }

    TinyIndex layerCount() { return static_cast<uint_fast8_t >(layers.size()); }

    void setWeights( half_or_float const *const in );

    void setBiasWeights( half_or_float const *const in );

protected:
    bool dirty;
    size_t maxOutputSize;
    CudaContext::ptr ctx;
    LayerContainer layers;

};


#endif //VIZDOOM_NEURALNETWORK_H

//
// Created by deano on 28/04/16.
//

#ifndef VIZDOOM_NEURALNETWORK_H
#define VIZDOOM_NEURALNETWORK_H

#include <vector>
#include <forward_list>
#include "neurallayer.h"

/*
 * Holds a neural network, currently restrict to a single item per layer
 */
class NeuralNetwork {
public:
    typedef std::shared_ptr< NeuralNetwork > ptr;

    NeuralNetwork( const TinyIndex _depth ) { layers.resize(_depth); }

    using LayerContainer = std::vector< NeuralLayer::ptr >;

    LayerContainer::const_iterator cbegin() const { return layers.cbegin(); }

    LayerContainer::const_iterator cend() const { return layers.cend(); }

    LayerContainer::iterator begin() { return layers.begin(); }

    LayerContainer::iterator end() { return layers.end(); }

    const NeuralLayer::ptr &operator[]( const TinyIndex _layer ) const {
        return layers.at(_layer);
    }

    NeuralLayer::ptr &operator[]( const TinyIndex _layer ) {
        return layers.at(_layer);
    }

    TinyIndex layerCount() { return static_cast<uint_fast8_t >(layers.size()); }

protected:
    LayerContainer layers;

};


#endif //VIZDOOM_NEURALNETWORK_H

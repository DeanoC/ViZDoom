//
// Created by deano on 28/04/16.
//

#ifndef VIZDOOM_NEURALLAYER_H
#define VIZDOOM_NEURALLAYER_H

#include "common.h"
#include <memory>

class NeuralLayer {
public:
    typedef std::shared_ptr< NeuralLayer > ptr;

    virtual~NeuralLayer() { }

    virtual void forwardPropogate( half_or_float const alpha, half_or_float const beta, half_or_float const *const x,
                                   half_or_float *y ) = 0;

    virtual void backPropogate( const half_or_float alpha, const half_or_float beta, const half_or_float *const x,
                                half_or_float *y ) = 0;

    virtual size_t getInputCount() const = 0;

    virtual size_t getWeightCount() const = 0;

    virtual size_t getOutputCount() const = 0;

    virtual void setWeights( half_or_float const *const in ) = 0;

    virtual bool hasBias() = 0;

    virtual void setBiasWeights( half_or_float const *const in ) = 0;

};


#endif //VIZDOOM_NEURALLAYER_H

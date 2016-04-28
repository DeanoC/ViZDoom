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

    virtual void forwardPropogate( const float alpha, const float beta, const float *x, float *y ) = 0;

    virtual void backPropogate( const float alpha, const float beta, const float *x, float *y ) = 0;

};


#endif //VIZDOOM_NEURALLAYER_H

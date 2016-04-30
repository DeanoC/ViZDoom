//
// Created by deano on 28/04/16.
//

#include "common.h"
#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork( CudaContext::ptr _context, const TinyIndex _depth ) :
        dirty(true),
        maxOutputSize(0),
        ctx(_context),
        layers(_depth, nullptr) {
}

size_t NeuralNetwork::getWeightCount() const {
    size_t sum = 0;
    for( auto &&layer : layers ) {
        sum += layer->getWeightCount();
    }
    return sum;
}

size_t NeuralNetwork::getBiasWeightCount() const {
    size_t sum = 0;
    for( auto &&layer : layers ) {
        sum += layer->hasBias() ? layer->getWeightCount() : 0;
    }
    return sum;
}

void NeuralNetwork::setWeights( half_or_float const *const in ) {
    int i = 0;
    for( auto &&layer : layers ) {
        layer->setWeights(&in[ i ]);
        i += layer->getWeightCount();
    }
}

void NeuralNetwork::setBiasWeights( half_or_float const *const in ) {
    int i = 0;
    for( auto &&layer : layers ) {
        if( layer->hasBias() ) {
            layer->setBiasWeights(&in[ i ]);
            i += layer->getOutputCount();
        }
    }
}

void NeuralNetwork::presentInput( half_or_float const *const input, half_or_float *output ) {
    if( dirty ) {
        maxOutputSize = 0;
        for( auto &&layer : layers ) {
            maxOutputSize = std::max(layer->getOutputCount() * sizeof(half_or_float), maxOutputSize);
        }
        ctx->reserveWorkspace(2 * maxOutputSize);
        dirty = false;
    }

    half_or_float const *in = input;
    half_or_float *out[2] = {
            (half_or_float *) ctx->grabWorkspace(maxOutputSize),
            (half_or_float *) ctx->grabWorkspace(maxOutputSize)
    };
    int db = 0;

    for( auto &&layer : layers ) {
        layer->forwardPropogate(half_or_float(1.0), half_or_float(0.0), in, out[ db ]);
        in = out[ db ];
        db = (db + 1) & 0x1;
    }

    // todo remove this copy for last layer if possible
    // in is actually our actual out at this point
    cudaMemcpy(output, in, layers.back()->getOutputCount(), cudaMemcpyKind::cudaMemcpyDefault);

    ctx->releaseWorkspace(out[ 1 ], maxOutputSize);
    ctx->releaseWorkspace(out[ 0 ], maxOutputSize);

}

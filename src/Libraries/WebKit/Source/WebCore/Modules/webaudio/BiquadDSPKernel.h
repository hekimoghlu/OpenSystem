/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include "AudioDSPKernel.h"
#include "Biquad.h"
#include "BiquadProcessor.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class BiquadProcessor;

// BiquadDSPKernel is an AudioDSPKernel and is responsible for filtering one channel of a BiquadProcessor using a Biquad object.

class BiquadDSPKernel final : public AudioDSPKernel {
    WTF_MAKE_TZONE_ALLOCATED(BiquadDSPKernel);
public:
    explicit BiquadDSPKernel(BiquadProcessor* processor)
    : AudioDSPKernel(processor)
    {
    }
    
    // AudioDSPKernel
    void process(std::span<const float> source, std::span<float> destination) final;
    void reset() override { m_biquad.reset(); }

    // Get the magnitude and phase response of the filter at the given
    // set of frequencies (in Hz). The phase response is in radians.
    void getFrequencyResponse(unsigned nFrequencies, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse);

    double tailTime() const override;
    double latencyTime() const override;

    void updateCoefficients(size_t numberOfFrames, std::span<const float> cutoffFrequency, std::span<const float> q, std::span<const float> gain, std::span<const float> detune);

    bool requiresTailProcessing() const final;

private:
    Biquad m_biquad;
    BiquadProcessor* biquadProcessor() { return downcast<BiquadProcessor>(processor()); }

    // To prevent audio glitches when parameters are changed,
    // dezippering is used to slowly change the parameters.
    // |useSmoothing| implies that we want to update using the
    // smoothed values. Otherwise the final target values are
    // used. If |forceUpdate| is true, we update the coefficients even
    // if they are not dirty. (Used when computing the frequency
    // response.)
    void updateCoefficientsIfNecessary(size_t framesToProcess);
    void updateTailTime(size_t coefIndex);

    double m_tailTime { std::numeric_limits<double>::infinity() };
};

} // namespace WebCore

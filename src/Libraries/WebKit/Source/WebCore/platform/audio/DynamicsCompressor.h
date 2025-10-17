/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#ifndef DynamicsCompressor_h
#define DynamicsCompressor_h

#include "DynamicsCompressorKernel.h"
#include "ZeroPole.h"
#include <memory>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueArray.h>

namespace WebCore {

class AudioBus;

// DynamicsCompressor implements a flexible audio dynamics compression effect such as
// is commonly used in musical production and game audio. It lowers the volume
// of the loudest parts of the signal and raises the volume of the softest parts,
// making the sound richer, fuller, and more controlled.

class DynamicsCompressor final {
    WTF_MAKE_TZONE_ALLOCATED(DynamicsCompressor);
    WTF_MAKE_NONCOPYABLE(DynamicsCompressor);
public:
    enum {
        ParamThreshold,
        ParamKnee,
        ParamRatio,
        ParamAttack,
        ParamRelease,
        ParamPreDelay,
        ParamReleaseZone1,
        ParamReleaseZone2,
        ParamReleaseZone3,
        ParamReleaseZone4,
        ParamPostGain,
        ParamEffectBlend,
        ParamReduction,
        ParamLast
    };

    DynamicsCompressor(float sampleRate, unsigned numberOfChannels);

    void process(const AudioBus* sourceBus, AudioBus* destinationBus, unsigned framesToProcess);
    void reset();
    void setNumberOfChannels(unsigned);

    void setParameterValue(unsigned parameterID, float value);
    float parameterValue(unsigned parameterID);

    float sampleRate() const { return m_sampleRate; }
    float nyquist() const { return m_sampleRate / 2; }

    double tailTime() const { return m_compressor.tailTime(); }
    double latencyTime() const { return m_compressor.latencyFrames() / static_cast<double>(sampleRate()); }
    bool requiresTailProcessing() const
    {
        // Always return true even if the tail time and latency might both be zero.
        return true;
    }

protected:
    unsigned m_numberOfChannels;

    // m_parameters holds the tweakable compressor parameters.
    std::array<float, ParamLast> m_parameters;
    void initializeParameters();

    std::span<std::span<const float>> sourceChannels() const { return unsafeMakeSpan(m_sourceChannels.get(), m_numberOfChannels); }
    std::span<std::span<float>> destinationChannels() const { return unsafeMakeSpan(m_destinationChannels.get(), m_numberOfChannels); }

    float m_sampleRate;

    UniqueArray<std::span<const float>> m_sourceChannels;
    UniqueArray<std::span<float>> m_destinationChannels;

    // The core compressor.
    DynamicsCompressorKernel m_compressor;
};

} // namespace WebCore

#endif // DynamicsCompressor_h

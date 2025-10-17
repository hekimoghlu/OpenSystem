/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#include "config.h"

#if ENABLE(WEB_AUDIO)

#include "DynamicsCompressor.h"

#include "AudioBus.h"
#include "AudioUtilities.h"
#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DynamicsCompressor);

using namespace AudioUtilities;
    
DynamicsCompressor::DynamicsCompressor(float sampleRate, unsigned numberOfChannels)
    : m_numberOfChannels(numberOfChannels)
    , m_sampleRate(sampleRate)
    , m_compressor(sampleRate, numberOfChannels)
{
    setNumberOfChannels(numberOfChannels);
    initializeParameters();
}

void DynamicsCompressor::setParameterValue(unsigned parameterID, float value)
{
    ASSERT(parameterID < ParamLast);
    if (parameterID < ParamLast)
        m_parameters[parameterID] = value;
}

void DynamicsCompressor::initializeParameters()
{
    // Initializes compressor to default values.
    
    m_parameters[ParamThreshold] = -24; // dB
    m_parameters[ParamKnee] = 30; // dB
    m_parameters[ParamRatio] = 12; // unit-less
    m_parameters[ParamAttack] = 0.003f; // seconds
    m_parameters[ParamRelease] = 0.250f; // seconds
    m_parameters[ParamPreDelay] = 0.006f; // seconds

    // Release zone values 0 -> 1.
    m_parameters[ParamReleaseZone1] = 0.09f;
    m_parameters[ParamReleaseZone2] = 0.16f;
    m_parameters[ParamReleaseZone3] = 0.42f;
    m_parameters[ParamReleaseZone4] = 0.98f;
    
    m_parameters[ParamPostGain] = 0; // dB
    m_parameters[ParamReduction] = 0; // dB

    // Linear crossfade (0 -> 1).
    m_parameters[ParamEffectBlend] = 1;
}

float DynamicsCompressor::parameterValue(unsigned parameterID)
{
    ASSERT(parameterID < ParamLast);
    return m_parameters[parameterID];
}

void DynamicsCompressor::process(const AudioBus* sourceBus, AudioBus* destinationBus, unsigned framesToProcess)
{
    // Though numberOfChannels is retrived from destinationBus, we still name it numberOfChannels instead of numberOfDestinationChannels.
    // It's because we internally match sourceChannels's size to destinationBus by channel up/down mix. Thus we need numberOfChannels
    // to do the loop work for both m_sourceChannels and m_destinationChannels.

    unsigned numberOfChannels = destinationBus->numberOfChannels();
    unsigned numberOfSourceChannels = sourceBus->numberOfChannels();

    ASSERT(numberOfChannels == m_numberOfChannels && numberOfSourceChannels);

    if (numberOfChannels != m_numberOfChannels || !numberOfSourceChannels) {
        destinationBus->zero();
        return;
    }

    switch (numberOfChannels) {
    case 2: // stereo
        m_sourceChannels[0] = sourceBus->channel(0)->span();

        if (numberOfSourceChannels > 1)
            m_sourceChannels[1] = sourceBus->channel(1)->span();
        else
            // Simply duplicate mono channel input data to right channel for stereo processing.
            m_sourceChannels[1] = m_sourceChannels[0];

        break;
    default:
        // FIXME : support other number of channels.
        ASSERT_NOT_REACHED();
        destinationBus->zero();
        return;
    }

    for (unsigned i = 0; i < numberOfChannels; ++i)
        m_destinationChannels[i] = destinationBus->channel(i)->mutableSpan();

    float dbThreshold = parameterValue(ParamThreshold);
    float dbKnee = parameterValue(ParamKnee);
    float ratio = parameterValue(ParamRatio);
    float attackTime = parameterValue(ParamAttack);
    float releaseTime = parameterValue(ParamRelease);
    float preDelayTime = parameterValue(ParamPreDelay);

    // This is effectively a master volume on the compressed signal (pre-blending).
    float dbPostGain = parameterValue(ParamPostGain);

    // Linear blending value from dry to completely processed (0 -> 1)
    // 0 means the signal is completely unprocessed.
    // 1 mixes in only the compressed signal.
    float effectBlend = parameterValue(ParamEffectBlend);

    float releaseZone1 = parameterValue(ParamReleaseZone1);
    float releaseZone2 = parameterValue(ParamReleaseZone2);
    float releaseZone3 = parameterValue(ParamReleaseZone3);
    float releaseZone4 = parameterValue(ParamReleaseZone4);

    // Apply compression to the source signal.
    m_compressor.process(sourceChannels(),
                         destinationChannels(),
                         framesToProcess,
                         dbThreshold,
                         dbKnee,
                         ratio,
                         attackTime,
                         releaseTime,
                         preDelayTime,
                         dbPostGain,
                         effectBlend,

                         releaseZone1,
                         releaseZone2,
                         releaseZone3,
                         releaseZone4
                         );
                         
    // Update the compression amount.                     
    setParameterValue(ParamReduction, m_compressor.meteringGain());
}

void DynamicsCompressor::reset()
{
    m_compressor.reset();
}

void DynamicsCompressor::setNumberOfChannels(unsigned numberOfChannels)
{
    m_sourceChannels = makeUniqueArray<std::span<const float>>(numberOfChannels);
    m_destinationChannels = makeUniqueArray<std::span<float>>(numberOfChannels);

    m_compressor.setNumberOfChannels(numberOfChannels);
    m_numberOfChannels = numberOfChannels;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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

#include "BiquadProcessor.h"

#include "AudioUtilities.h"
#include "BiquadDSPKernel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BiquadProcessor);

BiquadProcessor::BiquadProcessor(BaseAudioContext& context, float sampleRate, size_t numberOfChannels, bool autoInitialize)
    : AudioDSPKernelProcessor(sampleRate, numberOfChannels)
    , m_parameter1(AudioParam::create(context, "frequency"_s, 350.0, 0.0, 0.5 * sampleRate, AutomationRate::ARate))
    , m_parameter2(AudioParam::create(context, "Q"_s, 1, -FLT_MAX, FLT_MAX, AutomationRate::ARate))
    , m_parameter3(AudioParam::create(context, "gain"_s, 0.0, -FLT_MAX, 40 * std::log10(std::numeric_limits<float>::max()), AutomationRate::ARate))
    , m_parameter4(AudioParam::create(context, "detune"_s, 0.0, -153600, 153600, AutomationRate::ARate))
{
    if (autoInitialize)
        initialize();
}

BiquadProcessor::~BiquadProcessor()
{
    if (isInitialized())
        uninitialize();
}

std::unique_ptr<AudioDSPKernel> BiquadProcessor::createKernel()
{
    return makeUnique<BiquadDSPKernel>(this);
}

void BiquadProcessor::checkForDirtyCoefficients()
{
    // Deal with smoothing / de-zippering. Start out assuming filter parameters are not changing.

    // The BiquadDSPKernel objects rely on this value to see if they need to re-compute their internal filter coefficients.
    m_filterCoefficientsDirty = false;
    m_hasSampleAccurateValues = false;
    
    if (m_parameter1->hasSampleAccurateValues() || m_parameter2->hasSampleAccurateValues() || m_parameter3->hasSampleAccurateValues() || m_parameter4->hasSampleAccurateValues()) {
        m_filterCoefficientsDirty = true;
        m_hasSampleAccurateValues = true;

        m_shouldUseARate = m_parameter1->automationRate() == AutomationRate::ARate
            || m_parameter2->automationRate() == AutomationRate::ARate
            || m_parameter3->automationRate() == AutomationRate::ARate
            || m_parameter4->automationRate() == AutomationRate::ARate;
    } else {
        if (m_hasJustReset) {
            // Snap to exact values first time after reset, then smooth for subsequent changes.
            m_parameter1->resetSmoothedValue();
            m_parameter2->resetSmoothedValue();
            m_parameter3->resetSmoothedValue();
            m_parameter4->resetSmoothedValue();
            m_filterCoefficientsDirty = true;
            m_hasJustReset = false;
        } else {
            // Smooth all of the filter parameters. If they haven't yet converged to their target value then mark coefficients as dirty.
            bool isStable1 = m_parameter1->smooth();
            bool isStable2 = m_parameter2->smooth();
            bool isStable3 = m_parameter3->smooth();
            bool isStable4 = m_parameter4->smooth();
            if (!(isStable1 && isStable2 && isStable3 && isStable4))
                m_filterCoefficientsDirty = true;
        }
    }
}

void BiquadProcessor::process(const AudioBus* source, AudioBus* destination, size_t framesToProcess)
{
    if (!isInitialized()) {
        destination->zero();
        return;
    }
        
    checkForDirtyCoefficients();
            
    // For each channel of our input, process using the corresponding BiquadDSPKernel into the output channel.
    for (unsigned i = 0; i < m_kernels.size(); ++i)
        m_kernels[i]->process(source->channel(i)->span().first(framesToProcess), destination->channel(i)->mutableSpan());
}

void BiquadProcessor::processOnlyAudioParams(size_t framesToProcess)
{
    std::array<float, AudioUtilities::renderQuantumSize> values;
    ASSERT(framesToProcess <= AudioUtilities::renderQuantumSize);

    auto valuesSpan = std::span { values }.first(framesToProcess);
    m_parameter1->calculateSampleAccurateValues(valuesSpan);
    m_parameter2->calculateSampleAccurateValues(valuesSpan);
    m_parameter3->calculateSampleAccurateValues(valuesSpan);
    m_parameter4->calculateSampleAccurateValues(valuesSpan);
}

void BiquadProcessor::setType(BiquadFilterType type)
{
    if (type != m_type) {
        m_type = type;
        reset(); // The filter state must be reset only if the type has changed.
    }
}

void BiquadProcessor::getFrequencyResponse(unsigned nFrequencies, std::span<const float> frequencyHz, std::span<float> magResponse, std::span<float> phaseResponse)
{
    // Compute the frequency response on a separate temporary kernel
    // to avoid interfering with the processing running in the audio
    // thread on the main kernels.
    
    auto responseKernel = makeUnique<BiquadDSPKernel>(this);

    // Get a copy of the current biquad filter coefficients so we can update
    // |responseKernel| with these values.
    float cutoffFrequency = parameter1().value();
    float q = parameter2().value();
    float gain = parameter3().value();
    float detune = parameter4().value();

    responseKernel->updateCoefficients(1, singleElementSpan(cutoffFrequency), singleElementSpan(q), singleElementSpan(gain), singleElementSpan(detune));
    responseKernel->getFrequencyResponse(nFrequencies, frequencyHz, magResponse, phaseResponse);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

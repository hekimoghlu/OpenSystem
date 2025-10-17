/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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

#include "DynamicsCompressorNode.h"

#include "AudioContext.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include "AudioUtilities.h"
#include "DynamicsCompressor.h"
#include <wtf/TZoneMallocInlines.h>

// Set output to stereo by default.
static constexpr unsigned defaultNumberOfOutputChannels = 2;

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DynamicsCompressorNode);

ExceptionOr<Ref<DynamicsCompressorNode>> DynamicsCompressorNode::create(BaseAudioContext& context, const DynamicsCompressorOptions& options)
{
    auto node = adoptRef(*new DynamicsCompressorNode(context, options));

    auto result = node->handleAudioNodeOptions(options, { 2, ChannelCountMode::ClampedMax, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();

    return node;
}

DynamicsCompressorNode::DynamicsCompressorNode(BaseAudioContext& context, const DynamicsCompressorOptions& options)
    : AudioNode(context, NodeTypeDynamicsCompressor)
    , m_threshold(AudioParam::create(context, "threshold"_s, options.threshold, -100, 0, AutomationRate::KRate, AutomationRateMode::Fixed))
    , m_knee(AudioParam::create(context, "knee"_s, options.knee, 0, 40, AutomationRate::KRate, AutomationRateMode::Fixed))
    , m_ratio(AudioParam::create(context, "ratio"_s, options.ratio, 1, 20, AutomationRate::KRate, AutomationRateMode::Fixed))
    , m_attack(AudioParam::create(context, "attack"_s, options.attack, 0, 1, AutomationRate::KRate, AutomationRateMode::Fixed))
    , m_release(AudioParam::create(context, "release"_s, options.release, 0, 1, AutomationRate::KRate, AutomationRateMode::Fixed))
{
    addInput();
    addOutput(defaultNumberOfOutputChannels);

    initialize();
}

DynamicsCompressorNode::~DynamicsCompressorNode()
{
    uninitialize();
}

void DynamicsCompressorNode::process(size_t framesToProcess)
{
    AudioBus* outputBus = output(0)->bus();
    ASSERT(outputBus);

    float threshold = m_threshold->finalValue();
    float knee = m_knee->finalValue();
    float ratio = m_ratio->finalValue();
    float attack = m_attack->finalValue();
    float release = m_release->finalValue();

    m_dynamicsCompressor->setParameterValue(DynamicsCompressor::ParamThreshold, threshold);
    m_dynamicsCompressor->setParameterValue(DynamicsCompressor::ParamKnee, knee);
    m_dynamicsCompressor->setParameterValue(DynamicsCompressor::ParamRatio, ratio);
    m_dynamicsCompressor->setParameterValue(DynamicsCompressor::ParamAttack, attack);
    m_dynamicsCompressor->setParameterValue(DynamicsCompressor::ParamRelease, release);

    m_dynamicsCompressor->process(input(0)->bus(), outputBus, framesToProcess);

    setReduction(m_dynamicsCompressor->parameterValue(DynamicsCompressor::ParamReduction));
}

void DynamicsCompressorNode::processOnlyAudioParams(size_t framesToProcess)
{
    std::array<float, AudioUtilities::renderQuantumSize> values;
    ASSERT(framesToProcess <= AudioUtilities::renderQuantumSize);

    auto valuesSpan = std::span { values }.first(framesToProcess);
    m_threshold->calculateSampleAccurateValues(valuesSpan);
    m_knee->calculateSampleAccurateValues(valuesSpan);
    m_ratio->calculateSampleAccurateValues(valuesSpan);
    m_attack->calculateSampleAccurateValues(valuesSpan);
    m_release->calculateSampleAccurateValues(valuesSpan);
}

void DynamicsCompressorNode::initialize()
{
    if (isInitialized())
        return;

    AudioNode::initialize();    
    m_dynamicsCompressor = makeUnique<DynamicsCompressor>(sampleRate(), defaultNumberOfOutputChannels);
}

void DynamicsCompressorNode::uninitialize()
{
    if (!isInitialized())
        return;

    m_dynamicsCompressor = nullptr;
    AudioNode::uninitialize();
}

double DynamicsCompressorNode::tailTime() const
{
    return m_dynamicsCompressor->tailTime();
}

double DynamicsCompressorNode::latencyTime() const
{
    return m_dynamicsCompressor->latencyTime();
}

bool DynamicsCompressorNode::requiresTailProcessing() const
{
    // Always return true even if the tail time and latency might both be zero.
    return true;
}

ExceptionOr<void> DynamicsCompressorNode::setChannelCount(unsigned count)
{
    if (count > 2)
        return Exception { ExceptionCode::NotSupportedError, "DynamicsCompressorNode's channel count cannot be greater than 2"_s };
    return AudioNode::setChannelCount(count);
}

ExceptionOr<void> DynamicsCompressorNode::setChannelCountMode(ChannelCountMode mode)
{
    if (mode == ChannelCountMode::Max)
        return Exception { ExceptionCode::NotSupportedError, "DynamicsCompressorNode's channel count mode cannot be set to 'max'"_s };
    return AudioNode::setChannelCountMode(mode);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

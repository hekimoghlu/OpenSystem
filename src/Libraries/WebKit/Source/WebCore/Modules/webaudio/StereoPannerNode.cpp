/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "StereoPannerNode.h"

#if ENABLE(WEB_AUDIO)

#include "AudioBus.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include "AudioUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StereoPannerNode);

ExceptionOr<Ref<StereoPannerNode>> StereoPannerNode::create(BaseAudioContext& context, const StereoPannerOptions& options)
{
    auto stereo = adoptRef(*new StereoPannerNode(context, options.pan));
    
    auto result = stereo->handleAudioNodeOptions(options, { 2, ChannelCountMode::ClampedMax, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();
    
    return stereo;
}

StereoPannerNode::StereoPannerNode(BaseAudioContext& context, float pan)
    : AudioNode(context, NodeTypeStereoPanner)
    , m_pan(AudioParam::create(context, "pan"_s, pan, -1, 1, AutomationRate::ARate))
    , m_sampleAccurateValues(AudioUtilities::renderQuantumSize)
{
    addInput();
    addOutput(2);
    
    initialize();
}

StereoPannerNode::~StereoPannerNode()
{
    uninitialize();
}

void StereoPannerNode::process(size_t framesToProcess)
{
    AudioBus* destination = output(0)->bus();
    
    if (!isInitialized() || !input(0)->isConnected()) {
        destination->zero();
        return;
    }
    
    AudioBus* source = input(0)->bus();
    if (!source) {
        destination->zero();
        return;
    }

    if (m_pan->hasSampleAccurateValues() && m_pan->automationRate() == AutomationRate::ARate) {
        auto panValues = m_sampleAccurateValues.span().first(framesToProcess);
        m_pan->calculateSampleAccurateValues(panValues);
        StereoPanner::panWithSampleAccurateValues(source, destination, panValues);
        return;
    }
    
    // The pan value is not sample-accurate or not a-rate. In this case, we have
    // a fixed pan value for the render and just need to incorporate any inputs to
    // the value, if any.
    float panValue = m_pan->hasSampleAccurateValues() ? m_pan->finalValue() : m_pan->value();
    StereoPanner::panToTargetValue(source, destination, panValue, framesToProcess);
}

void StereoPannerNode::processOnlyAudioParams(size_t framesToProcess)
{
    std::array<float, AudioUtilities::renderQuantumSize> values;
    ASSERT(framesToProcess <= AudioUtilities::renderQuantumSize);

    m_pan->calculateSampleAccurateValues(std::span { values }.first(framesToProcess));
}

ExceptionOr<void> StereoPannerNode::setChannelCount(unsigned channelCount)
{
    if (channelCount > 2)
        return Exception { ExceptionCode::NotSupportedError, "StereoPannerNode's channelCount cannot be greater than 2."_s };
    
    return AudioNode::setChannelCount(channelCount);
}

ExceptionOr<void> StereoPannerNode::setChannelCountMode(ChannelCountMode mode)
{
    if (mode == ChannelCountMode::Max)
        return Exception { ExceptionCode::NotSupportedError, "StereoPannerNode's channelCountMode cannot be max."_s };
    
    return AudioNode::setChannelCountMode(mode);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

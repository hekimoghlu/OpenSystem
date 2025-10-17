/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "ConstantSourceNode.h"

#include "AudioNodeOutput.h"
#include "AudioParam.h"
#include "AudioUtilities.h"
#include "ConstantSourceOptions.h"
#include <algorithm>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ConstantSourceNode);

ExceptionOr<Ref<ConstantSourceNode>> ConstantSourceNode::create(BaseAudioContext& context, const ConstantSourceOptions& options)
{
    auto node = adoptRef(*new ConstantSourceNode(context, options.offset));
    node->suspendIfNeeded();
    return node;
}

ConstantSourceNode::ConstantSourceNode(BaseAudioContext& context, float offset)
    : AudioScheduledSourceNode(context, NodeTypeConstant)
    , m_offset(AudioParam::create(context, "offset"_s, offset, -FLT_MAX, FLT_MAX, AutomationRate::ARate))
    , m_sampleAccurateValues(AudioUtilities::renderQuantumSize)
{
    addOutput(1);
    initialize();
}

ConstantSourceNode::~ConstantSourceNode()
{
    uninitialize();
}

void ConstantSourceNode::process(size_t framesToProcess)
{
    auto& outputBus = *output(0)->bus();
    
    if (!isInitialized() || !outputBus.numberOfChannels()) {
        outputBus.zero();
        return;
    }
    
    size_t quantumFrameOffset = 0;
    size_t nonSilentFramesToProcess = 0;
    double startFrameOffset = 0;
    updateSchedulingInfo(framesToProcess, outputBus, quantumFrameOffset, nonSilentFramesToProcess, startFrameOffset);
    
    if (!nonSilentFramesToProcess) {
        outputBus.zero();
        return;
    }
    
    bool isSampleAccurate = m_offset->hasSampleAccurateValues();
    if (isSampleAccurate && m_offset->automationRate() == AutomationRate::ARate) {
        auto offsets = m_sampleAccurateValues.span();
        m_offset->calculateSampleAccurateValues(offsets.first(framesToProcess));
        if (nonSilentFramesToProcess > 0) {
            memcpySpan(outputBus.channel(0)->mutableSpan().subspan(quantumFrameOffset), offsets.subspan(quantumFrameOffset, nonSilentFramesToProcess));
            outputBus.clearSilentFlag();
        } else
            outputBus.zero();
        return;
    }
    
    float value = isSampleAccurate ? m_offset->finalValue() : m_offset->value();
    if (!value)
        outputBus.zero();
    else {
        auto destination = outputBus.channel(0)->mutableSpan();
        std::ranges::fill(destination.subspan(quantumFrameOffset).first(nonSilentFramesToProcess), value);
        outputBus.clearSilentFlag();
    }
}

bool ConstantSourceNode::propagatesSilence() const
{
    return !isPlayingOrScheduled() || hasFinished();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

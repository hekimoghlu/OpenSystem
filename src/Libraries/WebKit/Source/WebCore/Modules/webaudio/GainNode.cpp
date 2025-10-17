/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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

#include "GainNode.h"

#include "AudioBus.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include "AudioUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(GainNode);

ExceptionOr<Ref<GainNode>> GainNode::create(BaseAudioContext& context, const GainOptions& options)
{
    auto gainNode = adoptRef(*new GainNode(context));

    auto result = gainNode->handleAudioNodeOptions(options, { 2, ChannelCountMode::Max, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();

    gainNode->gain().setValue(options.gain);

    return gainNode;
}

GainNode::GainNode(BaseAudioContext& context)
    : AudioNode(context, NodeTypeGain)
    , m_sampleAccurateGainValues(AudioUtilities::renderQuantumSize) // FIXME: can probably share temp buffer in context
    , m_gain(AudioParam::create(context, "gain"_s, 1.0, -FLT_MAX, FLT_MAX, AutomationRate::ARate))
{
    addInput();
    addOutput(1);

    initialize();
}

void GainNode::process(size_t framesToProcess)
{
    // FIXME: for some cases there is a nice optimization to avoid processing here, and let the gain change
    // happen in the summing junction input of the AudioNode we're connected to.
    // Then we can avoid all of the following:

    AudioBus* outputBus = output(0)->bus();
    ASSERT(outputBus);

    if (!isInitialized() || !input(0)->isConnected())
        outputBus->zero();
    else {
        AudioBus* inputBus = input(0)->bus();

        if (gain().hasSampleAccurateValues() && gain().automationRate() == AutomationRate::ARate) {
            // Apply sample-accurate gain scaling for precise envelopes, grain windows, etc.
            ASSERT(framesToProcess <= m_sampleAccurateGainValues.size());
            if (framesToProcess <= m_sampleAccurateGainValues.size()) {
                auto gainValues = m_sampleAccurateGainValues.span().first(framesToProcess);
                gain().calculateSampleAccurateValues(gainValues);
                outputBus->copyWithSampleAccurateGainValuesFrom(*inputBus, gainValues);
            }
        } else {
            // Apply the gain with de-zippering into the output bus.
            float gain = this->gain().hasSampleAccurateValues() ? this->gain().finalValue() : this->gain().value();
            if (!gain) {
                // If the gain is 0 just zero the bus.
                outputBus->zero();
            } else
                outputBus->copyWithGainFrom(*inputBus, gain);
        }
    }
}

void GainNode::processOnlyAudioParams(size_t framesToProcess)
{
    std::array<float, AudioUtilities::renderQuantumSize> values;
    ASSERT(framesToProcess <= AudioUtilities::renderQuantumSize);

    m_gain->calculateSampleAccurateValues(std::span { values }.first(framesToProcess));
}

// FIXME: this can go away when we do mixing with gain directly in summing junction of AudioNodeInput
//
// As soon as we know the channel count of our input, we can lazily initialize.
// Sometimes this may be called more than once with different channel counts, in which case we must safely
// uninitialize and then re-initialize with the new channel count.
void GainNode::checkNumberOfChannelsForInput(AudioNodeInput* input)
{
    ASSERT(context().isAudioThread() && context().isGraphOwner());

    ASSERT(input && input == this->input(0));
    if (input != this->input(0))
        return;
        
    unsigned numberOfChannels = input->numberOfChannels();    

    if (isInitialized() && numberOfChannels != output(0)->numberOfChannels()) {
        // We're already initialized but the channel count has changed.
        uninitialize();
    }

    if (!isInitialized()) {
        // This will propagate the channel count to any nodes connected further downstream in the graph.
        output(0)->setNumberOfChannels(numberOfChannels);
        initialize();
    }

    AudioNode::checkNumberOfChannelsForInput(input);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

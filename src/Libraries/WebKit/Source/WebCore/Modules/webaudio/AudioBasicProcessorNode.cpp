/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#include "AudioBasicProcessorNode.h"

#include "AudioBus.h"
#include "AudioContext.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include "AudioProcessor.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioBasicProcessorNode);

AudioBasicProcessorNode::AudioBasicProcessorNode(BaseAudioContext& context, NodeType type)
    : AudioNode(context, type)
{
    addInput();
    addOutput(1);

    // The subclass must create m_processor.
}

AudioBasicProcessorNode::~AudioBasicProcessorNode() = default;

void AudioBasicProcessorNode::initialize()
{
    if (isInitialized())
        return;

    ASSERT(processor());
    processor()->initialize();

    AudioNode::initialize();
}

void AudioBasicProcessorNode::uninitialize()
{
    if (!isInitialized())
        return;

    ASSERT(processor());
    processor()->uninitialize();

    AudioNode::uninitialize();
}

void AudioBasicProcessorNode::process(size_t framesToProcess)
{
    AudioBus* destinationBus = output(0)->bus();
    
    if (!isInitialized() || !processor() || processor()->numberOfChannels() != numberOfChannels())
        destinationBus->zero();
    else {
        AudioBus* sourceBus = input(0)->bus();

        // FIXME: if we take "tail time" into account, then we can avoid calling processor()->process() once the tail dies down.
        if (!input(0)->isConnected())
            sourceBus->zero();

        processor()->process(sourceBus, destinationBus, framesToProcess);  
    }
}

void AudioBasicProcessorNode::processOnlyAudioParams(size_t framesToProcess)
{
    if (!isInitialized() || !processor())
        return;

    processor()->processOnlyAudioParams(framesToProcess);
}

// Nice optimization in the very common case allowing for "in-place" processing
void AudioBasicProcessorNode::pullInputs(size_t framesToProcess)
{
    // Render input stream - suggest to the input to render directly into output bus for in-place processing in process() if possible.
    input(0)->pull(output(0)->bus(), framesToProcess);
}

// As soon as we know the channel count of our input, we can lazily initialize.
// Sometimes this may be called more than once with different channel counts, in which case we must safely
// uninitialize and then re-initialize with the new channel count.
void AudioBasicProcessorNode::checkNumberOfChannelsForInput(AudioNodeInput* input)
{
    ASSERT(context().isAudioThread() && context().isGraphOwner());
    
    ASSERT(input == this->input(0));
    if (input != this->input(0))
        return;

    ASSERT(processor());
    if (!processor())
        return;

    unsigned numberOfChannels = input->numberOfChannels();
    
    if (isInitialized() && numberOfChannels != output(0)->numberOfChannels()) {
        // We're already initialized but the channel count has changed.
        uninitialize();
    }
    
    if (!isInitialized()) {
        // This will propagate the channel count to any nodes connected further down the chain...
        output(0)->setNumberOfChannels(numberOfChannels);

        // Re-initialize the processor with the new channel count.
        processor()->setNumberOfChannels(numberOfChannels);
        initialize();
    }

    AudioNode::checkNumberOfChannelsForInput(input);
}

unsigned AudioBasicProcessorNode::numberOfChannels()
{
    return output(0)->numberOfChannels();
}

double AudioBasicProcessorNode::tailTime() const
{
    return m_processor->tailTime();
}

double AudioBasicProcessorNode::latencyTime() const
{
    return m_processor->latencyTime();
}

bool AudioBasicProcessorNode::requiresTailProcessing() const
{
    return m_processor->requiresTailProcessing();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

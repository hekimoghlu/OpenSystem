/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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

#include "ChannelMergerNode.h"

#include "AudioContext.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ChannelMergerNode);

ExceptionOr<Ref<ChannelMergerNode>> ChannelMergerNode::create(BaseAudioContext& context, const ChannelMergerOptions& options)
{
    if (options.numberOfInputs > AudioContext::maxNumberOfChannels || !options.numberOfInputs)
        return Exception { ExceptionCode::IndexSizeError, "Number of inputs is not in the allowed range."_s };
    
    auto merger = adoptRef(*new ChannelMergerNode(context, options.numberOfInputs));
    
    auto result = merger->handleAudioNodeOptions(options, { 1, ChannelCountMode::Explicit, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();
    
    return merger;
}

ChannelMergerNode::ChannelMergerNode(BaseAudioContext& context, unsigned numberOfInputs)
    : AudioNode(context, NodeTypeChannelMerger)
{
    // Create the requested number of inputs.
    for (unsigned i = 0; i < numberOfInputs; ++i)
        addInput();

    addOutput(numberOfInputs);
    
    initialize();

    Locker contextLocker { context.graphLock() };
    disableOutputs();
}

void ChannelMergerNode::process(size_t framesToProcess)
{
    AudioNodeOutput* output = this->output(0);
    ASSERT(output);
    ASSERT_UNUSED(framesToProcess, framesToProcess == output->bus()->length());
    ASSERT(numberOfInputs() == output->numberOfChannels());
    
    // Merge all the channels from all the inputs into one output.
    for (unsigned i = 0; i < numberOfInputs(); ++i) {
        AudioNodeInput* input = this->input(i);
        ASSERT(input->numberOfChannels() == 1u);
        auto* outputChannel = output->bus()->channel(i);
        if (input->isConnected()) {
            // The mixing rules will be applied so multiple channels are down-
            // mixed to mono (when the mixing rule is defined). Note that only
            // the first channel will be taken for the undefined input channel
            // layout.
            //
            // See:
            // http://webaudio.github.io/web-audio-api/#channel-up-mixing-and-down-mixing
            auto* inputChannel = input->bus()->channel(0);
            outputChannel->copyFrom(inputChannel);
        } else {
            // If input is unconnected, fill zeros in the channel.
            outputChannel->zero();
        }
    }
}

ExceptionOr<void> ChannelMergerNode::setChannelCount(unsigned channelCount)
{
    if (channelCount != 1)
        return Exception { ExceptionCode::InvalidStateError, "Channel count cannot be changed from 1."_s };
    
    return AudioNode::setChannelCount(channelCount);
}

ExceptionOr<void> ChannelMergerNode::setChannelCountMode(ChannelCountMode mode)
{
    if (mode != ChannelCountMode::Explicit)
        return Exception { ExceptionCode::InvalidStateError, "Channel count mode cannot be changed from explicit."_s };
    
    return AudioNode::setChannelCountMode(mode);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

#include "ChannelSplitterNode.h"

#include "AudioContext.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ChannelSplitterNode);

ExceptionOr<Ref<ChannelSplitterNode>> ChannelSplitterNode::create(BaseAudioContext& context, const ChannelSplitterOptions& options)
{
    if (options.numberOfOutputs > AudioContext::maxNumberOfChannels || !options.numberOfOutputs)
        return Exception { ExceptionCode::IndexSizeError, "Number of outputs is not in the allowed range"_s };
    
    auto splitter = adoptRef(*new ChannelSplitterNode(context, options.numberOfOutputs));
    
    auto result = splitter->handleAudioNodeOptions(options, { options.numberOfOutputs, ChannelCountMode::Explicit, ChannelInterpretation::Discrete });
    if (result.hasException())
        return result.releaseException();
    
    return splitter;
}

ChannelSplitterNode::ChannelSplitterNode(BaseAudioContext& context, unsigned numberOfOutputs)
    : AudioNode(context, NodeTypeChannelSplitter)
{
    addInput();

    // Create a fixed number of outputs (able to handle the maximum number of channels fed to an input).
    for (unsigned i = 0; i < numberOfOutputs; ++i)
        addOutput(1);
    
    initialize();
}

void ChannelSplitterNode::process(size_t framesToProcess)
{
    AudioBus* source = input(0)->bus();
    ASSERT(source);
    ASSERT_UNUSED(framesToProcess, framesToProcess == source->length());
    
    unsigned numberOfSourceChannels = source->numberOfChannels();
    
    for (unsigned i = 0; i < numberOfOutputs(); ++i) {
        AudioBus* destination = output(i)->bus();
        ASSERT(destination);
        
        if (i < numberOfSourceChannels) {
            // Split the channel out if it exists in the source.
            // It would be nice to avoid the copy and simply pass along pointers, but this becomes extremely difficult with fanout and fanin.
            destination->channel(0)->copyFrom(source->channel(i));
        } else if (output(i)->renderingFanOutCount() > 0) {
            // Only bother zeroing out the destination if it's connected to anything
            destination->zero();
        }
    }
}

ExceptionOr<void> ChannelSplitterNode::setChannelCount(unsigned channelCount)
{
    if (channelCount != numberOfOutputs())
        return Exception { ExceptionCode::InvalidStateError, "Channel count must be set to number of outputs."_s };
    
    return AudioNode::setChannelCount(channelCount);
}

ExceptionOr<void> ChannelSplitterNode::setChannelCountMode(ChannelCountMode mode)
{
    if (mode != ChannelCountMode::Explicit)
        return Exception { ExceptionCode::InvalidStateError, "Channel count mode cannot be changed from explicit."_s };
    
    return AudioNode::setChannelCountMode(mode);
}

ExceptionOr<void> ChannelSplitterNode::setChannelInterpretation(ChannelInterpretation interpretation)
{
    if (interpretation != ChannelInterpretation::Discrete)
        return Exception { ExceptionCode::InvalidStateError, "Channel interpretation cannot be changed from discrete."_s };
    
    return AudioNode::setChannelInterpretation(interpretation);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

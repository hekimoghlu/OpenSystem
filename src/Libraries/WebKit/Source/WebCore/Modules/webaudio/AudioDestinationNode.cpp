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

#include "AudioDestinationNode.h"

#include "AudioBus.h"
#include "AudioContext.h"
#include "AudioIOCallback.h"
#include "AudioNodeInput.h"
#include "AudioNodeOutput.h"
#include "AudioUtilities.h"
#include "AudioWorklet.h"
#include "AudioWorkletGlobalScope.h"
#include "AudioWorkletMessagingProxy.h"
#include "AudioWorkletThread.h"
#include "DenormalDisabler.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
    
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioDestinationNode);

AudioDestinationNode::AudioDestinationNode(BaseAudioContext& context, float sampleRate)
    : AudioNode(context, NodeTypeDestination)
    , m_sampleRate(sampleRate)
{
    addInput();
}

AudioDestinationNode::~AudioDestinationNode()
{
    uninitialize();
}

void AudioDestinationNode::renderQuantum(AudioBus* destinationBus, size_t numberOfFrames, const AudioIOPosition& outputPosition)
{
    // We don't want denormals slowing down any of the audio processing
    // since they can very seriously hurt performance.
    // This will take care of all AudioNodes because they all process within this scope.
    DenormalDisabler denormalDisabler;
    
    context().setAudioThread(Thread::current());

    // For performance reasons, we forbid heap allocations while doing rendering on the audio thread.
    // Heap allocations that cannot be avoided or have not been fixed yet can be allowed using
    // DisableMallocRestrictionsForCurrentThreadScope scope variables.
    ForbidMallocUseForCurrentThreadScope forbidMallocUse;
    
    if (!context().isInitialized()) {
        destinationBus->zero();
        return;
    }

    ASSERT(numberOfFrames);
    if (!numberOfFrames) {
        destinationBus->zero();
        return;
    }

    // Let the context take care of any business at the start of each render quantum.
    context().handlePreRenderTasks(outputPosition);

    RefPtr<AudioWorkletGlobalScope> workletGlobalScope;
    if (RefPtr audioWorkletProxy = context().audioWorklet().proxy()) {
        if (Ref workletThread = audioWorkletProxy->workletThread(); workletThread->thread() == &Thread::current())
            workletGlobalScope = workletThread->globalScope();
    }
    if (workletGlobalScope)
        workletGlobalScope->handlePreRenderTasks();

    // This will cause the node(s) connected to us to process, which in turn will pull on their input(s),
    // all the way backwards through the rendering graph.
    AudioBus* renderedBus = input(0)->pull(destinationBus, numberOfFrames);

    if (!renderedBus)
        destinationBus->zero();
    else if (renderedBus != destinationBus) {
        // in-place processing was not possible - so copy
        destinationBus->copyFrom(*renderedBus);
    }

    // Process nodes which need a little extra help because they are not connected to anything, but still need to process.
    context().processAutomaticPullNodes(numberOfFrames);

    // Let the context take care of any business at the end of each render quantum.
    context().handlePostRenderTasks();
    
    // Advance current sample-frame.
    m_currentSampleFrame += numberOfFrames;

    if (workletGlobalScope)
        workletGlobalScope->handlePostRenderTasks(m_currentSampleFrame);
}

void AudioDestinationNode::ref() const
{
    context().ref();
}

void AudioDestinationNode::deref() const
{
    context().deref();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

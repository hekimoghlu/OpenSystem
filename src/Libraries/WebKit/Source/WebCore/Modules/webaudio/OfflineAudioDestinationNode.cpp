/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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

#include "OfflineAudioDestinationNode.h"

#include "AudioBuffer.h"
#include "AudioBus.h"
#include "AudioContext.h"
#include "AudioUtilities.h"
#include "AudioWorklet.h"
#include "AudioWorkletMessagingProxy.h"
#include "HRTFDatabaseLoader.h"
#include "OfflineAudioContext.h"
#include "WorkerRunLoop.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/JSGenericTypedArrayViewInlines.h>
#include <algorithm>
#include <wtf/MainThread.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/threads/BinarySemaphore.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(OfflineAudioDestinationNode);

OfflineAudioDestinationNode::OfflineAudioDestinationNode(OfflineAudioContext& context, unsigned numberOfChannels, float sampleRate, RefPtr<AudioBuffer>&& renderTarget)
    : AudioDestinationNode(context, sampleRate)
    , m_numberOfChannels(numberOfChannels)
    , m_renderTarget(WTFMove(renderTarget))
    , m_renderBus(AudioBus::create(numberOfChannels, AudioUtilities::renderQuantumSize))
    , m_framesToProcess(m_renderTarget ? m_renderTarget->length() : 0)
{
    initializeDefaultNodeOptions(numberOfChannels, ChannelCountMode::Explicit, ChannelInterpretation::Speakers);
}

OfflineAudioDestinationNode::~OfflineAudioDestinationNode()
{
    uninitialize();
}

OfflineAudioContext& OfflineAudioDestinationNode::context()
{
    return downcast<OfflineAudioContext>(AudioDestinationNode::context());
}

const OfflineAudioContext& OfflineAudioDestinationNode::context() const
{
    return downcast<OfflineAudioContext>(AudioDestinationNode::context());
}

void OfflineAudioDestinationNode::initialize()
{
    if (isInitialized())
        return;

    AudioNode::initialize();
}

void OfflineAudioDestinationNode::uninitialize()
{
    if (!isInitialized())
        return;

    if (m_startedRendering) {
        if (m_renderThread) {
            m_renderThread->waitForCompletion();
            m_renderThread = nullptr;
        }
        if (RefPtr workletProxy = context().audioWorklet().proxy()) {
            BinarySemaphore semaphore;
            workletProxy->postTaskForModeToWorkletGlobalScope([&semaphore](ScriptExecutionContext&) mutable {
                semaphore.signal();
            }, WorkerRunLoop::defaultMode());
            semaphore.wait();
        }
    }

    AudioNode::uninitialize();
}

void OfflineAudioDestinationNode::startRendering(CompletionHandler<void(std::optional<Exception>&&)>&& completionHandler)
{
    ALWAYS_LOG(LOGIDENTIFIER);

    ASSERT(isMainThread());
    ASSERT(m_renderTarget.get());
    if (!m_renderTarget.get())
        return completionHandler(Exception { ExceptionCode::InvalidStateError, "OfflineAudioContextNode has no rendering buffer"_s });
    
    if (m_startedRendering)
        return completionHandler(Exception { ExceptionCode::InvalidStateError, "Already started rendering"_s });

    m_startedRendering = true;
    Ref protectedThis { *this };

    auto offThreadRendering = [this, protectedThis = WTFMove(protectedThis)]() mutable {
        auto result = renderOnAudioThread();
        callOnMainThread([this, result, currentSampleFrame = this->currentSampleFrame(), protectedThis = WTFMove(protectedThis)]() mutable {
            context().postTask([this, protectedThis = WTFMove(protectedThis), result, currentSampleFrame]() mutable {
                m_startedRendering = false;
                switch (result) {
                case RenderResult::Failure:
                    context().finishedRendering(false);
                    break;
                case RenderResult::Complete:
                    context().finishedRendering(true);
                    break;
                case RenderResult::Suspended:
                    context().didSuspendRendering(currentSampleFrame);
                    break;
                }
            });
        });
    };

    if (RefPtr workletProxy = context().audioWorklet().proxy()) {
        workletProxy->postTaskForModeToWorkletGlobalScope([offThreadRendering = WTFMove(offThreadRendering)](ScriptExecutionContext&) mutable {
            offThreadRendering();
        }, WorkerRunLoop::defaultMode());
        return completionHandler(std::nullopt);
    }

    // FIXME: We should probably limit the number of threads we create for offline audio.
    m_renderThread = Thread::create("offline renderer"_s, WTFMove(offThreadRendering), ThreadType::Audio, Thread::QOS::Default);
    completionHandler(std::nullopt);
}

auto OfflineAudioDestinationNode::renderOnAudioThread() -> RenderResult
{
    ASSERT(!isMainThread());
    ASSERT(m_renderBus.get());

    if (!m_renderBus.get())
        return RenderResult::Failure;

    RELEASE_ASSERT(context().isInitialized());

    bool channelsMatch = m_renderBus->numberOfChannels() == m_renderTarget->numberOfChannels();
    ASSERT(channelsMatch);
    if (!channelsMatch)
        return RenderResult::Failure;

    bool isRenderBusAllocated = m_renderBus->length() >= AudioUtilities::renderQuantumSize;
    ASSERT(isRenderBusAllocated);
    if (!isRenderBusAllocated)
        return RenderResult::Failure;

    // Break up the render target into smaller "render quantize" sized pieces.
    // Render until we're finished.
    unsigned numberOfChannels = m_renderTarget->numberOfChannels();

    while (m_framesToProcess > 0) {
        if (context().shouldSuspend())
            return RenderResult::Suspended;

        // Render one render quantum.
        renderQuantum(m_renderBus.get(), AudioUtilities::renderQuantumSize, { });
        
        size_t framesAvailableToCopy = std::min(m_framesToProcess, AudioUtilities::renderQuantumSize);
        
        for (unsigned channelIndex = 0; channelIndex < numberOfChannels; ++channelIndex) {
            auto source = m_renderBus->channel(channelIndex)->span().first(framesAvailableToCopy);
            auto destination = m_renderTarget->channelData(channelIndex)->typedMutableSpan();
            memcpySpan(destination.subspan(m_destinationOffset), source);
        }
        
        m_destinationOffset += framesAvailableToCopy;
        m_framesToProcess -= framesAvailableToCopy;
    }

    return RenderResult::Complete;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "LibWebRTCAudioModule.h"

#if USE(LIBWEBRTC)

#include "LibWebRTCAudioFormat.h"
#include "Logging.h"

#if PLATFORM(COCOA)
#include "IncomingAudioMediaStreamTrackRendererUnit.h"
#endif

namespace WebCore {

LibWebRTCAudioModule::LibWebRTCAudioModule()
    : m_queue(WorkQueue::create("WebKitWebRTCAudioModule"_s, WorkQueue::QOS::UserInteractive))
    , m_logTimer(*this, &LibWebRTCAudioModule::logTimerFired)
{
    ASSERT(isMainThread());
}

LibWebRTCAudioModule::~LibWebRTCAudioModule()
{
}

int32_t LibWebRTCAudioModule::RegisterAudioCallback(webrtc::AudioTransport* audioTransport)
{
    RELEASE_LOG(WebRTC, "LibWebRTCAudioModule::RegisterAudioCallback %d", !!audioTransport);

    m_audioTransport = audioTransport;
    return 0;
}

int32_t LibWebRTCAudioModule::StartPlayout()
{
    RELEASE_LOG(WebRTC, "LibWebRTCAudioModule::StartPlayout %d", m_isPlaying);

    if (m_isPlaying)
        return 0;

    m_isPlaying = true;
    callOnMainThread([this, protectedThis = Ref { *this }] {
        m_logTimer.startRepeating(logTimerInterval);
    });

    m_queue->dispatch([this, protectedThis = Ref { *this }] {
        m_pollingTime = MonotonicTime::now();
#if PLATFORM(COCOA)
        m_currentAudioSampleCount = 0;
#endif
        pollAudioData();
    });
    return 0;
}

int32_t LibWebRTCAudioModule::StopPlayout()
{
    RELEASE_LOG(WebRTC, "LibWebRTCAudioModule::StopPlayout %d", m_isPlaying);

    m_isPlaying = false;
    callOnMainThread([this, protectedThis = Ref { *this }] {
        m_logTimer.stop();
    });
    return 0;
}

void LibWebRTCAudioModule::logTimerFired()
{
    RELEASE_LOG_IF(m_timeSpent, WebRTC, "LibWebRTCAudioModule::pollAudioData, polling took too much time: %d ms", m_timeSpent);
    m_timeSpent = 0;
}

// libwebrtc uses 10ms frames.
const unsigned frameLengthMs = 1000 * LibWebRTCAudioFormat::chunkSampleCount / LibWebRTCAudioFormat::sampleRate;
const unsigned pollInterval = LibWebRTCAudioModule::PollSamplesCount * frameLengthMs;
const unsigned channels = 2;

Seconds LibWebRTCAudioModule::computeDelayUntilNextPolling()
{
    auto now = MonotonicTime::now();
    auto delayUntilNextPolling = m_pollingTime + Seconds::fromMilliseconds(pollInterval) - now;
    if (delayUntilNextPolling.milliseconds() < 0) {
        m_timeSpent = (now - m_pollingTime).milliseconds();
        delayUntilNextPolling = 0_s;
    }
    m_pollingTime = now + delayUntilNextPolling;
    return delayUntilNextPolling;
}

void LibWebRTCAudioModule::pollAudioData()
{
    if (!m_isPlaying)
        return;

    Function<void()> nextPollFunction = [this, protectedThis = Ref { *this }] {
        pollAudioData();
    };

    {
        // For performance reasons, we forbid heap allocations while doing rendering on the webrtc audio thread.
        ForbidMallocUseForCurrentThreadScope forbidMallocUse;

        pollFromSource();
    }
    m_queue->dispatchAfter(computeDelayUntilNextPolling(), WTFMove(nextPollFunction));
}

void LibWebRTCAudioModule::pollFromSource()
{
    if (!m_audioTransport)
        return;

    for (unsigned i = 0; i < PollSamplesCount; i++) {
        int64_t elapsedTime = -1;
        int64_t ntpTime = -1;
        char data[LibWebRTCAudioFormat::sampleByteSize * channels * LibWebRTCAudioFormat::chunkSampleCount];
        m_audioTransport->PullRenderData(LibWebRTCAudioFormat::sampleByteSize * 8, LibWebRTCAudioFormat::sampleRate, channels, LibWebRTCAudioFormat::chunkSampleCount, data, &elapsedTime, &ntpTime);
#if PLATFORM(COCOA)
        if (m_isRenderingIncomingAudioCounter)
            m_incomingAudioMediaStreamTrackRendererUnit->newAudioChunkPushed(m_currentAudioSampleCount);
        m_currentAudioSampleCount += LibWebRTCAudioFormat::chunkSampleCount;
#endif
    }
}

#if PLATFORM(COCOA)
BaseAudioMediaStreamTrackRendererUnit& LibWebRTCAudioModule::incomingAudioMediaStreamTrackRendererUnit()
{
    if (!m_incomingAudioMediaStreamTrackRendererUnit)
        m_incomingAudioMediaStreamTrackRendererUnit = makeUniqueWithoutRefCountedCheck<IncomingAudioMediaStreamTrackRendererUnit>(*this);
    return *m_incomingAudioMediaStreamTrackRendererUnit;
}
#endif

} // namespace WebCore

#endif // USE(LIBWEBRTC)

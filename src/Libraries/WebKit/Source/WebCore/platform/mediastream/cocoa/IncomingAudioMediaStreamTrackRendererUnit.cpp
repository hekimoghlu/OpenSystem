/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include "IncomingAudioMediaStreamTrackRendererUnit.h"

#if ENABLE(MEDIA_STREAM) && USE(LIBWEBRTC)

#include "AudioMediaStreamTrackRenderer.h"
#include "AudioMediaStreamTrackRendererUnit.h"
#include "AudioSampleDataSource.h"
#include "LibWebRTCAudioFormat.h"
#include "LibWebRTCAudioModule.h"
#include "Logging.h"
#include "WebAudioBufferList.h"
#include <pal/avfoundation/MediaTimeAVFoundation.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WorkQueue.h>

#include <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IncomingAudioMediaStreamTrackRendererUnit);

IncomingAudioMediaStreamTrackRendererUnit::IncomingAudioMediaStreamTrackRendererUnit(LibWebRTCAudioModule& audioModule)
    : m_audioModule(audioModule)
    , m_queue(WorkQueue::create("IncomingAudioMediaStreamTrackRendererUnit"_s, WorkQueue::QOS::UserInitiated))
#if !RELEASE_LOG_DISABLED
    , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
#endif
{
}

IncomingAudioMediaStreamTrackRendererUnit::~IncomingAudioMediaStreamTrackRendererUnit()
{
    ASSERT(isMainThread());
}

void IncomingAudioMediaStreamTrackRendererUnit::addResetObserver(const String& deviceID, ResetObserver& observer)
{
    AudioMediaStreamTrackRendererUnit::singleton().addResetObserver(deviceID, observer);
}

std::pair<bool, Vector<Ref<AudioSampleDataSource>>> IncomingAudioMediaStreamTrackRendererUnit::addSourceToMixer(const String& deviceID, Ref<AudioSampleDataSource>&& source)
{
    assertIsMainThread();

    ASSERT(!deviceID.isEmpty());
    auto& mixer = m_mixers.ensure(deviceID, [] { return Mixer { }; }).iterator->value;

    ASSERT(!mixer.sources.contains(source.get()));
    bool shouldStart = mixer.sources.isEmpty();
    mixer.sources.add(WTFMove(source));
    return std::make_pair(shouldStart, copyToVector(mixer.sources));
}

void IncomingAudioMediaStreamTrackRendererUnit::addSource(const String& identifier, Ref<AudioSampleDataSource>&& source)
{
    AudioMediaStreamTrackRendererUnit::singleton().setLastDeviceUsed(identifier);

    String deviceID = AudioMediaStreamTrackRendererUnit::supportsPerDeviceRendering() ? identifier : AudioMediaStreamTrackRenderer::defaultDeviceID();

#if !RELEASE_LOG_DISABLED
    source->logger().logAlways(LogWebRTC, "IncomingAudioMediaStreamTrackRendererUnit::addSource ", source->logIdentifier());
#endif
    ASSERT(isMainThread());
#if !RELEASE_LOG_DISABLED
        if (!m_logger)
            m_logger = &source->logger();
#endif

    ASSERT(source->outputDescription());
    if (!source->outputDescription())
        return;
    auto outputDescription = *source->outputDescription();

    auto result = addSourceToMixer(deviceID, WTFMove(source));
    bool shouldStart = result.first;
    auto newSources = WTFMove(result.second);

    postTask([this, newSources = WTFMove(newSources), shouldStart, deviceID = deviceID.isolatedCopy(), outputDescription]() mutable {
        assertIsCurrent(m_queue.get());

        m_renderMixers.ensure(deviceID, [] { return RenderMixer { }; }).iterator->value.inputSources = WTFMove(newSources);

        if (!shouldStart)
            return;

        if (!m_outputStreamDescription) {
            m_outputStreamDescription = outputDescription;
            m_audioBufferList = makeUnique<WebAudioBufferList>(*m_outputStreamDescription);
            m_sampleCount = m_outputStreamDescription->sampleRate() / 100;
            m_audioBufferList->setSampleCount(m_sampleCount);
        }

        auto mixedSource = AudioSampleDataSource::create(m_outputStreamDescription->sampleRate() * 0.5, *this, LibWebRTCAudioModule::PollSamplesCount + 1);
        mixedSource->setInputFormat(outputDescription);
        mixedSource->setOutputFormat(outputDescription);

        m_renderMixers.ensure(deviceID, [] { return RenderMixer { }; }).iterator->value.mixedSource = mixedSource.ptr();

        callOnMainThread([protectedThis = Ref { *this }, source = WTFMove(mixedSource), deviceID = deviceID.isolatedCopy()]() mutable {
            assertIsMainThread();

            auto& mixer = protectedThis->m_mixers.ensure(deviceID, [] { return Mixer { }; }).iterator->value;
            mixer.registeredMixedSource = WTFMove(source);
            mixer.deviceID = WTFMove(deviceID);
            protectedThis->start(mixer);
        });
    });
}

std::pair<bool, Vector<Ref<AudioSampleDataSource>>> IncomingAudioMediaStreamTrackRendererUnit::removeSourceFromMixer(const String& deviceID, AudioSampleDataSource& source)
{
    assertIsMainThread();

    auto& mixer = m_mixers.ensure(deviceID, [] { return Mixer { }; }).iterator->value;

    ASSERT(mixer.sources.contains(source));
    bool shouldStop = !mixer.sources.isEmpty();
    mixer.sources.remove(WTFMove(source));
    shouldStop &= mixer.sources.isEmpty();
    return std::make_pair(shouldStop, copyToVector(mixer.sources));
}

void IncomingAudioMediaStreamTrackRendererUnit::removeSource(const String& identifier, AudioSampleDataSource& source)
{
    String deviceID = AudioMediaStreamTrackRendererUnit::supportsPerDeviceRendering() ? identifier : AudioMediaStreamTrackRenderer::defaultDeviceID();

#if !RELEASE_LOG_DISABLED
    source.logger().logAlways(LogWebRTC, "IncomingAudioMediaStreamTrackRendererUnit::removeSource ", source.logIdentifier());
#endif
    ASSERT(isMainThread());

    auto result = removeSourceFromMixer(deviceID, source);
    bool shouldStop = result.first;
    auto newSources = WTFMove(result.second);

    postTask([this, newSources = WTFMove(newSources), shouldStop, deviceID = deviceID.isolatedCopy()]() mutable {
        assertIsCurrent(m_queue.get());

        m_renderMixers.ensure(deviceID, [] { return RenderMixer { }; }).iterator->value.inputSources = WTFMove(newSources);
        if (!shouldStop)
            return;

        callOnMainThread([protectedThis = Ref { *this }, deviceID = deviceID.isolatedCopy()]() mutable {
            assertIsMainThread();

            auto& mixer = protectedThis->m_mixers.ensure(deviceID, [] { return Mixer { }; }).iterator->value;
            mixer.deviceID = WTFMove(deviceID);
            protectedThis->stop(mixer);
        });
    });
}

void IncomingAudioMediaStreamTrackRendererUnit::start(Mixer& mixer)
{
    RELEASE_LOG(WebRTC, "IncomingAudioMediaStreamTrackRendererUnit::start");
    ASSERT(isMainThread());

    AudioMediaStreamTrackRendererUnit::singleton().addSource(mixer.deviceID, *mixer.registeredMixedSource);
    m_audioModule.get()->startIncomingAudioRendering();
}

void IncomingAudioMediaStreamTrackRendererUnit::stop(Mixer& mixer)
{
    RELEASE_LOG(WebRTC, "IncomingAudioMediaStreamTrackRendererUnit::stop");
    ASSERT(isMainThread());

    AudioMediaStreamTrackRendererUnit::singleton().removeSource(mixer.deviceID, *mixer.registeredMixedSource);
    mixer.registeredMixedSource = nullptr;
    m_audioModule.get()->stopIncomingAudioRendering();
}

void IncomingAudioMediaStreamTrackRendererUnit::postTask(Function<void()>&& callback)
{
    m_queue->dispatch([protectedThis = Ref { *this }, callback = WTFMove(callback)]() mutable {
        callback();
    });
}

void IncomingAudioMediaStreamTrackRendererUnit::newAudioChunkPushed(uint64_t currentAudioSampleCount)
{
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    postTask([this, currentAudioSampleCount] {
        renderAudioChunk(currentAudioSampleCount);
    });
}

void IncomingAudioMediaStreamTrackRendererUnit::renderAudioChunk(uint64_t currentAudioSampleCount)
{
    assertIsCurrent(m_queue.get());

    uint64_t timeStamp = currentAudioSampleCount * m_outputStreamDescription->sampleRate() / LibWebRTCAudioFormat::sampleRate;

    for (auto& renderMixer : m_renderMixers.values()) {
        // Mix all sources.
        bool hasCopiedData = false;
        for (auto& source : renderMixer.inputSources) {
            if (source->pullAvailableSampleChunk(*m_audioBufferList, m_sampleCount, timeStamp, hasCopiedData ? AudioSampleDataSource::Mix : AudioSampleDataSource::Copy))
                hasCopiedData = true;
        }

        CMTime startTime = PAL::CMTimeMake(renderMixer.writeCount, m_outputStreamDescription->sampleRate());
        if (hasCopiedData)
            renderMixer.mixedSource->pushSamples(PAL::toMediaTime(startTime), *m_audioBufferList, m_sampleCount);
        renderMixer.writeCount += m_sampleCount;
    }
}

#if !RELEASE_LOG_DISABLED
const Logger& IncomingAudioMediaStreamTrackRendererUnit::logger() const
{
    return *m_logger;
}

WTFLogChannel& IncomingAudioMediaStreamTrackRendererUnit::logChannel() const
{
    return LogWebRTC;
}

uint64_t IncomingAudioMediaStreamTrackRendererUnit::logIdentifier() const
{
    return m_logIdentifier;
}
#endif

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(LIBWEBRTC)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include "AudioMediaStreamTrackRendererUnit.h"

#if ENABLE(MEDIA_STREAM)

#include "AudioMediaStreamTrackRenderer.h"
#include "AudioSampleDataSource.h"
#include "Logging.h"

namespace WebCore {

AudioMediaStreamTrackRendererUnit& AudioMediaStreamTrackRendererUnit::singleton()
{
    static LazyNeverDestroyed<std::unique_ptr<AudioMediaStreamTrackRendererUnit>> sharedUnit;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        sharedUnit.construct(std::unique_ptr<AudioMediaStreamTrackRendererUnit>(new AudioMediaStreamTrackRendererUnit()));
    });
    return *sharedUnit.get();
}

bool AudioMediaStreamTrackRendererUnit::supportsPerDeviceRendering()
{
#if !PLATFORM(IOS_FAMILY)
    return true;
#else
    return false;
#endif
}

AudioMediaStreamTrackRendererUnit::AudioMediaStreamTrackRendererUnit()
    : m_deleteUnitsTimer([] { AudioMediaStreamTrackRendererUnit::singleton().deleteUnitsIfPossible(); })
{
}

AudioMediaStreamTrackRendererUnit::~AudioMediaStreamTrackRendererUnit() = default;

void AudioMediaStreamTrackRendererUnit::setLastDeviceUsed(const String& deviceID)
{
    if (supportsPerDeviceRendering())
        return;

    UNUSED_PARAM(deviceID);
    Ref unit = ensureDeviceUnit(AudioMediaStreamTrackRenderer::defaultDeviceID());
    unit->setLastDeviceUsed(deviceID);
}

void AudioMediaStreamTrackRendererUnit::deleteUnitsIfPossible()
{
    assertIsMainThread();

    m_units.removeIf([] (auto& keyValue) {
        if (keyValue.value->isDefault() || keyValue.value->hasSources())
            return false;

        Ref unit = keyValue.value;
        unit->close();
        return true;
    });
}

Ref<AudioMediaStreamTrackRendererUnit::Unit> AudioMediaStreamTrackRendererUnit::ensureDeviceUnit(const String& identifier)
{
    String deviceID = supportsPerDeviceRendering() ? identifier : AudioMediaStreamTrackRenderer::defaultDeviceID();

    assertIsMainThread();

    return m_units.ensure(deviceID, [&deviceID] {
        return Unit::create(deviceID);
    }).iterator->value;
}

RefPtr<AudioMediaStreamTrackRendererUnit::Unit> AudioMediaStreamTrackRendererUnit::getDeviceUnit(const String& identifier)
{
    String deviceID = supportsPerDeviceRendering() ? identifier : AudioMediaStreamTrackRenderer::defaultDeviceID();

    assertIsMainThread();

    auto iterator = m_units.find(deviceID);
    if (iterator == m_units.end())
        return { };
    return iterator->value.ptr();
}

void AudioMediaStreamTrackRendererUnit::addSource(const String& deviceID, Ref<AudioSampleDataSource>&& source)
{
    setLastDeviceUsed(deviceID);

    Ref unit = ensureDeviceUnit(deviceID);
    unit->addSource(WTFMove(source));
}

void AudioMediaStreamTrackRendererUnit::removeSource(const String& deviceID, AudioSampleDataSource& source)
{
    assertIsMainThread();

    RefPtr unit = getDeviceUnit(deviceID);
    if (!unit)
        return;

    static constexpr Seconds deleteUnitDelay = 10_s;
    if (unit->removeSource(source) && !unit->isDefault())
        m_deleteUnitsTimer.startOneShot(deleteUnitDelay);
}

void AudioMediaStreamTrackRendererUnit::addResetObserver(const String& deviceID, ResetObserver& observer)
{
    Ref unit = ensureDeviceUnit(deviceID);
    unit->addResetObserver(observer);
}

void AudioMediaStreamTrackRendererUnit::retrieveFormatDescription(CompletionHandler<void(std::optional<CAAudioStreamDescription>)>&& callback)
{
    assertIsMainThread();

    Ref unit = ensureDeviceUnit(AudioMediaStreamTrackRenderer::defaultDeviceID());
    unit->retrieveFormatDescription(WTFMove(callback));
}

AudioMediaStreamTrackRendererUnit::Unit::Unit(const String& deviceID)
    : m_internalUnit(AudioMediaStreamTrackRendererInternalUnit::create(deviceID, *this))
    , m_isDefaultUnit(deviceID == AudioMediaStreamTrackRenderer::defaultDeviceID())
{
}

AudioMediaStreamTrackRendererUnit::Unit::~Unit()
{
    stop();
}

void AudioMediaStreamTrackRendererUnit::Unit::close()
{
    assertIsMainThread();
    m_internalUnit->close();
}

void AudioMediaStreamTrackRendererUnit::Unit::addSource(Ref<AudioSampleDataSource>&& source)
{
#if !RELEASE_LOG_DISABLED
    source->logger().logAlways(LogWebRTC, "AudioMediaStreamTrackRendererUnit::addSource ", source->logIdentifier());
#endif
    assertIsMainThread();

    ASSERT(!m_sources.contains(source.get()));
    bool shouldStart = m_sources.isEmpty();
    m_sources.add(WTFMove(source));

    {
        Locker locker { m_pendingRenderSourcesLock };
        m_pendingRenderSources = copyToVector(m_sources);
        m_hasPendingRenderSources = true;
    }

    if (shouldStart)
        start();
}

bool AudioMediaStreamTrackRendererUnit::Unit::removeSource(AudioSampleDataSource& source)
{
#if !RELEASE_LOG_DISABLED
    source.logger().logAlways(LogWebRTC, "AudioMediaStreamTrackRendererUnit::removeSource ", source.logIdentifier());
#endif
    assertIsMainThread();

    bool shouldStop = !m_sources.isEmpty();
    m_sources.remove(source);
    shouldStop &= m_sources.isEmpty();

    {
        Locker locker { m_pendingRenderSourcesLock };
        m_pendingRenderSources = copyToVector(m_sources);
        m_hasPendingRenderSources = true;
    }

    if (shouldStop)
        stop();
    return shouldStop;
}

void AudioMediaStreamTrackRendererUnit::Unit::addResetObserver(ResetObserver& observer)
{
    assertIsMainThread();
    m_resetObservers.add(observer);
}

void AudioMediaStreamTrackRendererUnit::Unit::setLastDeviceUsed(const String& deviceID)
{
    assertIsMainThread();
    m_internalUnit->setLastDeviceUsed(deviceID);
}

void AudioMediaStreamTrackRendererUnit::Unit::retrieveFormatDescription(CompletionHandler<void(std::optional<CAAudioStreamDescription>)>&& callback)
{
    assertIsMainThread();
    m_internalUnit->retrieveFormatDescription(WTFMove(callback));
}

void AudioMediaStreamTrackRendererUnit::Unit::start()
{
    assertIsMainThread();
    RELEASE_LOG(WebRTC, "AudioMediaStreamTrackRendererUnit::start");

    m_internalUnit->start();
}

void AudioMediaStreamTrackRendererUnit::Unit::stop()
{
    assertIsMainThread();
    RELEASE_LOG(WebRTC, "AudioMediaStreamTrackRendererUnit::stop");

    m_internalUnit->stop();
}

void AudioMediaStreamTrackRendererUnit::Unit::reset()
{
    RELEASE_LOG(WebRTC, "AudioMediaStreamTrackRendererUnit::reset");
    if (!isMainThread()) {
        callOnMainThread([weakThis = ThreadSafeWeakPtr { *this }] {
            if (RefPtr strongThis = weakThis.get())
                strongThis->reset();
        });
        return;
    }

    assertIsMainThread();
    m_resetObservers.forEach([](auto& observer) {
        observer();
    });
}

void AudioMediaStreamTrackRendererUnit::Unit::updateRenderSourcesIfNecessary()
{
    if (!m_pendingRenderSourcesLock.tryLock())
        return;

    Locker locker { AdoptLock, m_pendingRenderSourcesLock };
    if (!m_hasPendingRenderSources)
        return;

    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    m_renderSources = WTFMove(m_pendingRenderSources);
    m_hasPendingRenderSources = false;
}

OSStatus AudioMediaStreamTrackRendererUnit::Unit::render(size_t sampleCount, AudioBufferList& ioData, uint64_t sampleTime, double hostTime, AudioUnitRenderActionFlags& actionFlags)
{
    // For performance reasons, we forbid heap allocations while doing rendering on the audio thread.
    ForbidMallocUseForCurrentThreadScope forbidMallocUse;

    ASSERT(!isMainThread());

    updateRenderSourcesIfNecessary();

    // Mix all sources.
    bool hasCopiedData = false;
    for (auto& source : m_renderSources) {
        if (source->pullSamples(ioData, sampleCount, sampleTime, hostTime, hasCopiedData ? AudioSampleDataSource::Mix : AudioSampleDataSource::Copy))
            hasCopiedData = true;
    }
    if (!hasCopiedData)
        actionFlags = kAudioUnitRenderAction_OutputIsSilence;
    return 0;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

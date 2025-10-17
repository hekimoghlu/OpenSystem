/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#include "MediaStreamPrivate.h"

#if ENABLE(MEDIA_STREAM)

#include "GraphicsContext.h"
#include "IntRect.h"
#include "Logging.h"
#include <wtf/Algorithms.h>
#include <wtf/MainThread.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

Ref<MediaStreamPrivate> MediaStreamPrivate::create(Ref<const Logger>&& logger, Ref<RealtimeMediaSource>&& source)
{
    auto loggerCopy = logger.copyRef();
    return MediaStreamPrivate::create(WTFMove(logger), MediaStreamTrackPrivateVector::from(MediaStreamTrackPrivate::create(WTFMove(loggerCopy), WTFMove(source))));
}

Ref<MediaStreamPrivate> MediaStreamPrivate::create(Ref<const Logger>&& logger, RefPtr<RealtimeMediaSource>&& audioSource, RefPtr<RealtimeMediaSource>&& videoSource)
{
    MediaStreamTrackPrivateVector tracks;
    tracks.reserveInitialCapacity(2);

    if (audioSource)
        tracks.append(MediaStreamTrackPrivate::create(logger.copyRef(), audioSource.releaseNonNull()));
    if (videoSource)
        tracks.append(MediaStreamTrackPrivate::create(logger.copyRef(), videoSource.releaseNonNull()));

    return MediaStreamPrivate::create(WTFMove(logger), tracks);
}

MediaStreamPrivate::MediaStreamPrivate(Ref<const Logger>&& logger, const MediaStreamTrackPrivateVector& tracks, String&& id)
    : m_id(WTFMove(id))
#if !RELEASE_LOG_DISABLED
    , m_logger(WTFMove(logger))
    , m_logIdentifier(uniqueLogIdentifier())
#endif
{
    UNUSED_PARAM(logger);
    ASSERT(!m_id.isEmpty());

    for (auto& track : tracks) {
        track->addObserver(*this);
        m_trackSet.add(track->id(), track);
    }
    m_isActive = computeActiveState();
    updateActiveVideoTrack();
}

MediaStreamPrivate::~MediaStreamPrivate()
{
    for (auto& track : m_trackSet.values())
        track->removeObserver(*this);
}

void MediaStreamPrivate::addObserver(MediaStreamPrivateObserver& observer)
{
    ASSERT(isMainThread());
    m_observers.add(observer);
}

void MediaStreamPrivate::removeObserver(MediaStreamPrivateObserver& observer)
{
    ASSERT(isMainThread());
    m_observers.remove(observer);
}

void MediaStreamPrivate::forEachObserver(const Function<void(MediaStreamPrivateObserver&)>& apply)
{
    ASSERT(isMainThread());
    Ref protectedThis { *this };
    m_observers.forEach(apply);
}

MediaStreamTrackPrivateVector MediaStreamPrivate::tracks() const
{
    return copyToVector(m_trackSet.values());
}

void MediaStreamPrivate::forEachTrack(const Function<void(const MediaStreamTrackPrivate&)>& callback) const
{
    for (auto& track : m_trackSet.values())
        callback(track.get());
}

void MediaStreamPrivate::forEachTrack(const Function<void(MediaStreamTrackPrivate&)>& callback)
{
    for (auto& track : m_trackSet.values())
        callback(track.get());
}

bool MediaStreamPrivate::computeActiveState()
{
    return WTF::anyOf(m_trackSet, [](auto& track) { return !track.value->ended(); });
}

void MediaStreamPrivate::updateActiveState()
{
    bool newActiveState = computeActiveState();

    updateActiveVideoTrack();

    // A stream is active if it has at least one un-ended track.
    if (newActiveState == m_isActive)
        return;

    m_isActive = newActiveState;

    forEachObserver([](auto& observer) {
        observer.activeStatusChanged();
    });
}

void MediaStreamPrivate::addTrack(Ref<MediaStreamTrackPrivate>&& track)
{
    if (m_trackSet.contains(track->id()))
        return;

    ALWAYS_LOG(LOGIDENTIFIER, track->logIdentifier());

    auto& trackRef = track.get();
    track->addObserver(*this);
    m_trackSet.add(track->id(), WTFMove(track));

    forEachObserver([&trackRef](auto& observer) {
        observer.didAddTrack(trackRef);
    });

    updateActiveState();
    characteristicsChanged();
}

void MediaStreamPrivate::removeTrack(MediaStreamTrackPrivate& track)
{
    if (!m_trackSet.remove(track.id()))
        return;

    ALWAYS_LOG(LOGIDENTIFIER, track.logIdentifier());
    track.removeObserver(*this);

    forEachObserver([&track](auto& observer) {
        observer.didRemoveTrack(track);
    });

    updateActiveState();
    characteristicsChanged();
}

void MediaStreamPrivate::startProducingData()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    for (auto& track : m_trackSet.values())
        track->startProducingData();
}

void MediaStreamPrivate::stopProducingData()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    for (auto& track : m_trackSet.values())
        track->stopProducingData();
}

bool MediaStreamPrivate::isProducingData() const
{
    for (auto& track : m_trackSet.values()) {
        if (track->isProducingData())
            return true;
    }
    return false;
}

bool MediaStreamPrivate::hasVideo() const
{
    for (auto& track : m_trackSet.values()) {
        if (track->isVideo() && track->isActive())
            return true;
    }
    return false;
}

bool MediaStreamPrivate::hasAudio() const
{
    for (auto& track : m_trackSet.values()) {
        if (track->isAudio() && track->isActive())
            return true;
    }
    return false;
}

bool MediaStreamPrivate::muted() const
{
    for (auto& track : m_trackSet.values()) {
        if (!track->muted() && !track->ended())
            return false;
    }
    return true;
}

IntSize MediaStreamPrivate::intrinsicSize() const
{
    IntSize size;
    if (m_activeVideoTrack) {
        const RealtimeMediaSourceSettings& setting = m_activeVideoTrack->settings();
        size.setWidth(setting.width());
        size.setHeight(setting.height());
    }

    return size;
}

void MediaStreamPrivate::updateActiveVideoTrack()
{
    m_activeVideoTrack = nullptr;
    for (auto& track : m_trackSet.values()) {
        if (!track->ended() && track->isVideo()) {
            m_activeVideoTrack = track.ptr();
            break;
        }
    }
}

void MediaStreamPrivate::characteristicsChanged()
{
    forEachObserver([](auto& observer) {
        observer.characteristicsChanged();
    });
}

void MediaStreamPrivate::trackMutedChanged(MediaStreamTrackPrivate& track)
{
#if RELEASE_LOG_DISABLED
    UNUSED_PARAM(track);
#endif

    ALWAYS_LOG(LOGIDENTIFIER, track.logIdentifier(), " ", track.muted());
    characteristicsChanged();
}

void MediaStreamPrivate::trackSettingsChanged(MediaStreamTrackPrivate&)
{
    characteristicsChanged();
}

void MediaStreamPrivate::trackEnabledChanged(MediaStreamTrackPrivate& track)
{
#if RELEASE_LOG_DISABLED
    UNUSED_PARAM(track);
#endif

    ALWAYS_LOG(LOGIDENTIFIER, track.logIdentifier(), " ", track.enabled());
    updateActiveVideoTrack();

    characteristicsChanged();
}

void MediaStreamPrivate::trackStarted(MediaStreamTrackPrivate& track)
{
#if RELEASE_LOG_DISABLED
    UNUSED_PARAM(track);
#endif

    ALWAYS_LOG(LOGIDENTIFIER, track.logIdentifier());
    characteristicsChanged();
}

void MediaStreamPrivate::trackEnded(MediaStreamTrackPrivate& track)
{
#if RELEASE_LOG_DISABLED
    UNUSED_PARAM(track);
#endif

    ALWAYS_LOG(LOGIDENTIFIER, track.logIdentifier());
    updateActiveState();
    characteristicsChanged();
}

void MediaStreamPrivate::monitorOrientation(OrientationNotifier& notifier)
{
    for (auto& track : m_trackSet.values()) {
        if (track->isCaptureTrack() && track->deviceType() == CaptureDevice::DeviceType::Camera)
            track->source().monitorOrientation(notifier);
    }
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& MediaStreamPrivate::logChannel() const
{
    return LogWebRTC;
}
#endif

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

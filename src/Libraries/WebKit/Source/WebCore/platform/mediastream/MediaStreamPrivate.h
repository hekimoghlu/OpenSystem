/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#ifndef MediaStreamPrivate_h
#define MediaStreamPrivate_h

#if ENABLE(MEDIA_STREAM)

#include "FloatSize.h"
#include "MediaStreamTrackPrivate.h"
#include <wtf/Function.h>
#include <wtf/MediaTime.h>
#include <wtf/RefPtr.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/UUID.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {
class MediaStreamPrivateObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MediaStreamPrivateObserver> : std::true_type { };
}

namespace WebCore {

class MediaStream;
class OrientationNotifier;

class MediaStreamPrivateObserver : public CanMakeWeakPtr<MediaStreamPrivateObserver> {
public:
    virtual ~MediaStreamPrivateObserver() = default;

    virtual void characteristicsChanged() { }
    virtual void activeStatusChanged() { }
    virtual void didAddTrack(MediaStreamTrackPrivate&) { }
    virtual void didRemoveTrack(MediaStreamTrackPrivate&) { }
};

class MediaStreamPrivate final
    : public MediaStreamTrackPrivateObserver
    , public RefCounted<MediaStreamPrivate>
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<MediaStreamPrivate> create(Ref<const Logger>&&, Ref<RealtimeMediaSource>&&);
    static Ref<MediaStreamPrivate> create(Ref<const Logger>&&, RefPtr<RealtimeMediaSource>&& audioSource, RefPtr<RealtimeMediaSource>&& videoSource);
    static Ref<MediaStreamPrivate> create(Ref<const Logger>&& logger, const MediaStreamTrackPrivateVector& tracks, String&& id = createVersion4UUIDString()) { return adoptRef(*new MediaStreamPrivate(WTFMove(logger), tracks, WTFMove(id))); }

    WEBCORE_EXPORT virtual ~MediaStreamPrivate();

    void addObserver(MediaStreamPrivateObserver&);
    void removeObserver(MediaStreamPrivateObserver&);

    String id() const { return m_id; }

    MediaStreamTrackPrivateVector tracks() const;
    bool hasTracks() const { return !m_trackSet.isEmpty(); }
    void forEachTrack(const Function<void(const MediaStreamTrackPrivate&)>&) const;
    void forEachTrack(const Function<void(MediaStreamTrackPrivate&)>&);
    MediaStreamTrackPrivate* activeVideoTrack() { return m_activeVideoTrack; }

    bool active() const { return m_isActive; }
    void updateActiveState();

    void addTrack(Ref<MediaStreamTrackPrivate>&&);
    WEBCORE_EXPORT void removeTrack(MediaStreamTrackPrivate&);

    void startProducingData();
    void stopProducingData();
    bool isProducingData() const;

    WEBCORE_EXPORT bool hasVideo() const;
    bool hasAudio() const;
    bool muted() const;

    IntSize intrinsicSize() const;

    void monitorOrientation(OrientationNotifier&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
#endif

private:
    MediaStreamPrivate(Ref<const Logger>&&, const MediaStreamTrackPrivateVector&, String&&);

    // MediaStreamTrackPrivateObserver
    void trackStarted(MediaStreamTrackPrivate&) override;
    void trackEnded(MediaStreamTrackPrivate&) override;
    void trackMutedChanged(MediaStreamTrackPrivate&) override;
    void trackSettingsChanged(MediaStreamTrackPrivate&) override;
    void trackEnabledChanged(MediaStreamTrackPrivate&) override;

    void characteristicsChanged();
    void updateActiveVideoTrack();

    bool computeActiveState();
    void forEachObserver(const Function<void(MediaStreamPrivateObserver&)>&);

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "MediaStreamPrivate"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    WeakHashSet<MediaStreamPrivateObserver> m_observers;
    String m_id;
    MediaStreamTrackPrivate* m_activeVideoTrack { nullptr };
    MemoryCompactRobinHoodHashMap<String, Ref<MediaStreamTrackPrivate>> m_trackSet;
    bool m_isActive { false };
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

#endif // MediaStreamPrivate_h

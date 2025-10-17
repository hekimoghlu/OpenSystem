/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#pragma once

#if ENABLE(MEDIA_STREAM)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "MediaCanStartListener.h"
#include "MediaProducer.h"
#include "MediaStreamPrivate.h"
#include "MediaStreamTrack.h"
#include "ScriptWrappable.h"
#include "Timer.h"
#include "URLRegistry.h"
#include <wtf/LoggerHelper.h>
#include <wtf/RefPtr.h>
#include <wtf/RobinHoodHashMap.h>

namespace WebCore {

class Document;

class MediaStream final
    : public EventTarget
    , public ActiveDOMObject
    , public MediaStreamPrivateObserver
    , private MediaCanStartListener
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
    , public RefCounted<MediaStream> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(MediaStream, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<MediaStream> create(Document&);
    static Ref<MediaStream> create(Document&, MediaStream&);
    static Ref<MediaStream> create(Document&, const Vector<Ref<MediaStreamTrack>>&);
    static Ref<MediaStream> create(Document&, Ref<MediaStreamPrivate>&&);
    WEBCORE_EXPORT virtual ~MediaStream();

    String id() const { return m_private->id(); }

    void addTrack(MediaStreamTrack&);
    void removeTrack(MediaStreamTrack&);
    MediaStreamTrack* getTrackById(String);

    MediaStreamTrack* getFirstAudioTrack() const;
    MediaStreamTrack* getFirstVideoTrack() const;

    MediaStreamTrackVector getAudioTracks() const;
    MediaStreamTrackVector getVideoTracks() const;
    MediaStreamTrackVector getTracks() const;

    RefPtr<MediaStream> clone();

    USING_CAN_MAKE_WEAKPTR(MediaStreamPrivateObserver);

    bool active() const { return m_isActive; }
    bool muted() const { return m_private->muted(); }

    template<typename Function> bool hasMatchingTrack(Function&& function) const { return anyOf(m_trackMap.values(), std::forward<Function>(function)); }

    MediaStreamPrivate& privateStream() { return m_private.get(); }
    Ref<MediaStreamPrivate> protectedPrivateStream();

    void startProducingData();
    void stopProducingData();

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::MediaStream; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

    void addTrackFromPlatform(Ref<MediaStreamTrack>&&);

#if !RELEASE_LOG_DISABLED
    uint64_t logIdentifier() const final { return m_private->logIdentifier(); }
#endif

protected:
    MediaStream(Document&, const Vector<Ref<MediaStreamTrack>>&);
    MediaStream(Document&, Ref<MediaStreamPrivate>&&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_private->logger(); }
    WTFLogChannel& logChannel() const final;
    ASCIILiteral logClassName() const final { return "MediaStream"_s; }
#endif

private:
    void internalAddTrack(Ref<MediaStreamTrack>&&);
    WEBCORE_EXPORT RefPtr<MediaStreamTrack> internalTakeTrack(const String&);

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // MediaStreamPrivateObserver
    void activeStatusChanged() final;
    void didAddTrack(MediaStreamTrackPrivate&) final;
    void didRemoveTrack(MediaStreamTrackPrivate&) final;
    void characteristicsChanged() final;

    MediaProducerMediaStateFlags mediaState() const;

    // MediaCanStartListener
    void mediaCanStart(Document&) final;

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    void updateActiveState();
    void activityEventTimerFired();
    void setIsActive(bool);
    void statusDidChange();

    MediaStreamTrackVector filteredTracks(const Function<bool(const MediaStreamTrack&)>&) const;

    Document* document() const;

    Ref<MediaStreamPrivate> m_private;

    MemoryCompactRobinHoodHashMap<String, Ref<MediaStreamTrack>> m_trackMap;

    MediaProducerMediaStateFlags m_state;

    bool m_isActive { false };
    bool m_isProducingData { false };
    bool m_isWaitingUntilMediaCanStart { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

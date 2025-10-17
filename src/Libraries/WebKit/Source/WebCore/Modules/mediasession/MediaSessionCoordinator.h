/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

#if ENABLE(MEDIA_SESSION_COORDINATOR)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "MediaSession.h"
#include "MediaSessionCoordinatorPrivate.h"
#include "MediaSessionCoordinatorState.h"
#include <wtf/Logger.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

template<typename> class DOMPromiseDeferred;

class MediaSessionCoordinator
    : public RefCounted<MediaSessionCoordinator>
    , public MediaSessionCoordinatorClient
    , public MediaSessionObserver
    , public ActiveDOMObject
    , public EventTarget  {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(MediaSessionCoordinator, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    WEBCORE_EXPORT static Ref<MediaSessionCoordinator> create(ScriptExecutionContext*);
    WEBCORE_EXPORT ~MediaSessionCoordinator();
    WEBCORE_EXPORT void setMediaSessionCoordinatorPrivate(Ref<MediaSessionCoordinatorPrivate>&&);

    void join(DOMPromiseDeferred<void>&&);
    ExceptionOr<void> leave();
    void close();

    String identifier() const { return m_privateCoordinator ? m_privateCoordinator->identifier() : String(); }
    MediaSessionCoordinatorState state() const { return m_state; }

    void seekTo(double, DOMPromiseDeferred<void>&&);
    void play(DOMPromiseDeferred<void>&&);
    void pause(DOMPromiseDeferred<void>&&);
    void setTrack(const String&, DOMPromiseDeferred<void>&&);

    void setMediaSession(MediaSession*);

    USING_CAN_MAKE_WEAKPTR(MediaSessionCoordinatorClient);

    struct PlaySessionCommand {
        std::optional<double> atTime;
        std::optional<MonotonicTime> hostTime;
    };
    std::optional<PlaySessionCommand> takeCurrentPlaySessionCommand() { return WTFMove(m_currentPlaySessionCommand); }

private:
    MediaSessionCoordinator(ScriptExecutionContext*);

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::MediaSessionCoordinator; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }
    void eventListenersDidChange() final;

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    // MediaSessionObserver
    void metadataChanged(const RefPtr<MediaMetadata>&) final;
    void positionStateChanged(const std::optional<MediaPositionState>&) final;
    void playbackStateChanged(MediaSessionPlaybackState) final;
    void readyStateChanged(MediaSessionReadyState) final;

    // MediaSessionCoordinatorClient
    void seekSessionToTime(double, CompletionHandler<void(bool)>&&) final;
    void playSession(std::optional<double> atTime, std::optional<MonotonicTime> hostTime, CompletionHandler<void(bool)>&&) final;
    void pauseSession(CompletionHandler<void(bool)>&&) final;
    void setSessionTrack(const String&, CompletionHandler<void(bool)>&&) final;
    void coordinatorStateChanged(WebCore::MediaSessionCoordinatorState) final;

    bool currentPositionApproximatelyEqualTo(double) const;

    const Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    static WTFLogChannel& logChannel();
    static ASCIILiteral logClassName() { return "MediaSessionCoordinator"_s; }
    bool shouldFireEvents() const;

    WeakPtr<MediaSession> m_session;
    RefPtr<MediaSessionCoordinatorPrivate> m_privateCoordinator;
    const Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
    MediaSessionCoordinatorState m_state { MediaSessionCoordinatorState::Closed };
    bool m_hasCoordinatorsStateChangeEventListener { false };
    std::optional<PlaySessionCommand> m_currentPlaySessionCommand;
};

}

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)

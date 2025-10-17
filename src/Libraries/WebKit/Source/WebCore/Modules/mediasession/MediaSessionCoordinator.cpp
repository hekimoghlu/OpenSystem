/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "MediaSessionCoordinator.h"

#if ENABLE(MEDIA_SESSION_COORDINATOR)

#include "Document.h"
#include "EventNames.h"
#include "JSDOMException.h"
#include "JSDOMPromiseDeferred.h"
#include "JSMediaSessionCoordinatorState.h"
#include "Logging.h"
#include "MediaMetadata.h"
#include "MediaSession.h"
#include "MediaSessionCoordinatorPrivate.h"
#include <wtf/CompletionHandler.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/Logger.h>
#include <wtf/LoggerHelper.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

static const Seconds CommandTimeTolerance = 50_ms;

namespace WebCore {

static uint64_t nextCoordinatorLogIdentifier()
{
    static uint64_t logIdentifier = cryptographicallyRandomNumber<uint32_t>();
    return ++logIdentifier;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaSessionCoordinator);

Ref<MediaSessionCoordinator> MediaSessionCoordinator::create(ScriptExecutionContext* context)
{
    auto coordinator = adoptRef(*new MediaSessionCoordinator(context));
    coordinator->suspendIfNeeded();
    return coordinator;
}

MediaSessionCoordinator::MediaSessionCoordinator(ScriptExecutionContext* context)
    : ActiveDOMObject(context)
    , m_logger(Document::sharedLogger())
    , m_logIdentifier(nextCoordinatorLogIdentifier())
{
    ALWAYS_LOG(LOGIDENTIFIER);
}

void MediaSessionCoordinator::setMediaSessionCoordinatorPrivate(Ref<MediaSessionCoordinatorPrivate>&& privateCoordinator)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    if (m_privateCoordinator)
        m_privateCoordinator->leave();
    m_privateCoordinator = WTFMove(privateCoordinator);
    m_privateCoordinator->setLogger(m_logger.copyRef(), m_logIdentifier);
    m_privateCoordinator->setClient(*this);
    coordinatorStateChanged(MediaSessionCoordinatorState::Waiting);
}

MediaSessionCoordinator::~MediaSessionCoordinator() = default;

void MediaSessionCoordinator::eventListenersDidChange()
{
    m_hasCoordinatorsStateChangeEventListener = hasEventListeners(eventNames().coordinatorstatechangeEvent);
}

bool MediaSessionCoordinator::virtualHasPendingActivity() const
{
    // Need to keep the JS wrapper alive as long as it may still fire events in the future.
    return shouldFireEvents();
}

void MediaSessionCoordinator::join(DOMPromiseDeferred<void>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier, m_state);

    if (m_state != MediaSessionCoordinatorState::Waiting) {
        ERROR_LOG(identifier, "invalid state");
        promise.reject(Exception { ExceptionCode::InvalidStateError, makeString("Unable to join when state is "_s, convertEnumerationToString(m_state)) });
        return;
    }
    ASSERT(m_privateCoordinator, "We must be in Waiting state if no private coordinator is set");

    m_privateCoordinator->join([protectedThis = Ref { *this }, identifier, promise = WTFMove(promise)] (std::optional<Exception>&& exception) mutable {
        if (!protectedThis->m_session) {
            promise.reject(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            protectedThis->logger().error(protectedThis->logChannel(), identifier, "coordinator.join failed!");
            promise.reject(WTFMove(*exception));
            return;
        }

        protectedThis->coordinatorStateChanged(MediaSessionCoordinatorState::Joined);

        promise.resolve();
    });
}

ExceptionOr<void> MediaSessionCoordinator::leave()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    if (m_state != MediaSessionCoordinatorState::Joined)
        return Exception { ExceptionCode::InvalidStateError, makeString("Unable to leave when state is "_s, convertEnumerationToString(m_state)) };

    close();

    return { };
}

void MediaSessionCoordinator::close()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    coordinatorStateChanged(MediaSessionCoordinatorState::Closed);
    if (!m_privateCoordinator)
        return;

    m_privateCoordinator->leave();
    m_privateCoordinator = nullptr;
}

void MediaSessionCoordinator::seekTo(double time, DOMPromiseDeferred<void>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier, time);

    if (!m_session) {
        ERROR_LOG(identifier, "MediaSession is NULL!");
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    if (m_state != MediaSessionCoordinatorState::Joined) {
        ERROR_LOG(identifier, ".state is ", m_state);
        promise.reject(Exception { ExceptionCode::InvalidStateError, makeString("Unable to seekTo when state is "_s, convertEnumerationToString(m_state)) });
        return;
    }

    m_privateCoordinator->seekTo(time, [protectedThis = Ref { *this }, identifier, promise = WTFMove(promise)] (std::optional<Exception>&& exception) mutable {
        if (!protectedThis->m_session) {
            promise.reject(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            promise.reject(WTFMove(*exception));
            protectedThis->logger().error(protectedThis->logChannel(), identifier, "coordinator.seekTo failed!");
            return;
        }

        promise.resolve();
    });
}

void MediaSessionCoordinator::play(DOMPromiseDeferred<void>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    if (!m_session) {
        ERROR_LOG(identifier, "MediaSession is NULL!");
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    if (m_state != MediaSessionCoordinatorState::Joined) {
        ERROR_LOG(identifier, ".state is ", m_state);
        promise.reject(Exception { ExceptionCode::InvalidStateError, makeString("Unable to play when state is "_s, convertEnumerationToString(m_state)) });
        return;
    }

    m_privateCoordinator->play([protectedThis = Ref { *this }, identifier, promise = WTFMove(promise)] (std::optional<Exception>&& exception) mutable {
        if (!protectedThis->m_session) {
            promise.reject(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            promise.reject(WTFMove(*exception));
            protectedThis->logger().error(protectedThis->logChannel(), identifier, "coordinator.play failed!");
            return;
        }

        promise.resolve();
    });
}

void MediaSessionCoordinator::pause(DOMPromiseDeferred<void>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    if (!m_session) {
        ERROR_LOG(identifier, "MediaSession is NULL!");
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    if (m_state != MediaSessionCoordinatorState::Joined) {
        ERROR_LOG(identifier, ".state is ", m_state);
        promise.reject(Exception { ExceptionCode::InvalidStateError, makeString("Unable to pause when state is "_s, convertEnumerationToString(m_state)) });
        return;
    }

    m_privateCoordinator->pause([protectedThis = Ref { *this }, identifier, promise = WTFMove(promise)] (std::optional<Exception>&& exception) mutable {
        if (!protectedThis->m_session) {
            promise.reject(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            promise.reject(WTFMove(*exception));
            protectedThis->logger().error(protectedThis->logChannel(), identifier, "coordinator.pause failed!");
            return;
        }

        promise.resolve();
    });
}

void MediaSessionCoordinator::setTrack(const String& track, DOMPromiseDeferred<void>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    if (!m_session) {
        ERROR_LOG(identifier, "MediaSession is NULL!");
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    if (m_state != MediaSessionCoordinatorState::Joined) {
        ERROR_LOG(identifier, ".state is ", m_state);
        promise.reject(Exception { ExceptionCode::InvalidStateError, makeString("Unable to setTrack when state is "_s, convertEnumerationToString(m_state)) });
        return;
    }

    m_privateCoordinator->setTrack(track, [protectedThis = Ref { *this }, identifier, promise = WTFMove(promise)] (std::optional<Exception>&& exception) mutable {
        if (!protectedThis->m_session) {
            promise.reject(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            promise.reject(WTFMove(*exception));
            protectedThis->logger().error(protectedThis->logChannel(), identifier, "coordinator.setTrack failed!");
            return;
        }

        promise.resolve();
    });
}

void MediaSessionCoordinator::setMediaSession(MediaSession* session)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    m_session = session;

    if (m_session)
        m_session->addObserver(*this);
}

void MediaSessionCoordinator::metadataChanged(const RefPtr<MediaMetadata>& metadata)
{
#if ENABLE(MEDIA_SESSION_PLAYLIST)
    if (!m_privateCoordinator)
        return;

    auto identifier = metadata ? metadata->trackIdentifier() : emptyString();
    ALWAYS_LOG(LOGIDENTIFIER, m_state, ", trackIdentifier:", identifier);
    m_privateCoordinator->trackIdentifierChanged(identifier);
#endif
}

void MediaSessionCoordinator::positionStateChanged(const std::optional<MediaPositionState>& positionState)
{
    if (positionState)
        ALWAYS_LOG(LOGIDENTIFIER, positionState.value());
    else
        ALWAYS_LOG(LOGIDENTIFIER, "{ }");

    if (m_state != MediaSessionCoordinatorState::Joined)
        return;

    if (!m_privateCoordinator)
        return;

    if (!positionState) {
        m_privateCoordinator->positionStateChanged({ });
        return;
    }

    m_privateCoordinator->positionStateChanged(MediaPositionState { positionState->duration, positionState->playbackRate, positionState->position });
}

void MediaSessionCoordinator::playbackStateChanged(MediaSessionPlaybackState playbackState)
{
    ALWAYS_LOG(LOGIDENTIFIER, m_state, ", ", playbackState);

    if (m_state != MediaSessionCoordinatorState::Joined)
        return;

    if (!m_privateCoordinator)
        return;

    m_privateCoordinator->playbackStateChanged(playbackState);
}

void MediaSessionCoordinator::readyStateChanged(MediaSessionReadyState readyState)
{
    ALWAYS_LOG(LOGIDENTIFIER, m_state, ", ", readyState);

    if (m_state != MediaSessionCoordinatorState::Joined)
        return;

    if (!m_privateCoordinator)
        return;

    m_privateCoordinator->readyStateChanged(readyState);
}

void MediaSessionCoordinator::seekSessionToTime(double time, CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG(LOGIDENTIFIER, m_state, ", ", time);

    if (m_state != MediaSessionCoordinatorState::Joined) {
        completionHandler(false);
        return;
    }

    bool isPaused = m_session->playbackState() == MediaSessionPlaybackState::Paused;

    if (isPaused && currentPositionApproximatelyEqualTo(time)) {
        completionHandler(true);
        return;
    }

    if (!isPaused)
        m_session->callActionHandler({ .action = MediaSessionAction::Pause });

    m_session->callActionHandler({ .action = MediaSessionAction::Seekto, .seekTime = time });
    completionHandler(true);
}

void MediaSessionCoordinator::playSession(std::optional<double> atTime, std::optional<MonotonicTime> hostTime, CompletionHandler<void(bool)>&& completionHandler)
{
    auto now = MonotonicTime::now();
    auto delta = hostTime ? *hostTime - now : Seconds::nan();
    ALWAYS_LOG(LOGIDENTIFIER, m_state, " time: ", atTime ? *atTime : -1, " hostTime: ", (hostTime ? *hostTime : MonotonicTime::nan()).secondsSinceEpoch().value(), " delta: ", delta.value());

    if (m_state != MediaSessionCoordinatorState::Joined) {
        completionHandler(false);
        return;
    }

    if (atTime && !currentPositionApproximatelyEqualTo(*atTime))
        m_session->callActionHandler({ .action = MediaSessionAction::Seekto, .seekTime = *atTime });

    m_currentPlaySessionCommand = { atTime, hostTime };
    m_session->callActionHandler({ .action = MediaSessionAction::Play });
    completionHandler(true);
}

void MediaSessionCoordinator::pauseSession(CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG(LOGIDENTIFIER, m_state);

    if (m_state != MediaSessionCoordinatorState::Joined) {
        completionHandler(false);
        return;
    }

    m_session->callActionHandler({ .action = MediaSessionAction::Pause });
    completionHandler(true);
}

void MediaSessionCoordinator::setSessionTrack(const String& track, CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG(LOGIDENTIFIER, m_state, ", ", track);

    if (m_state != MediaSessionCoordinatorState::Joined) {
        completionHandler(false);
        return;
    }

    m_session->callActionHandler({ .action = MediaSessionAction::Settrack, .trackIdentifier = track });
    completionHandler(true);
}

bool MediaSessionCoordinator::shouldFireEvents() const
{
    return m_hasCoordinatorsStateChangeEventListener && m_session;
}


void MediaSessionCoordinator::coordinatorStateChanged(MediaSessionCoordinatorState state)
{
    if (m_state == state)
        return;
    m_state = state;
    ALWAYS_LOG(LOGIDENTIFIER, m_state);
    if (shouldFireEvents())
        queueTaskToDispatchEvent(*this, TaskSource::MediaElement, Event::create(eventNames().coordinatorstatechangeEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

bool MediaSessionCoordinator::currentPositionApproximatelyEqualTo(double time) const
{
    if (!m_session)
        return false;

    auto currentPosition = m_session->currentPosition();
    if (!currentPosition)
        return false;

    auto delta = Seconds(std::abs(*currentPosition - time));
    return delta <= CommandTimeTolerance;
}

WTFLogChannel& MediaSessionCoordinator::logChannel()
{
    return LogMedia;
}

}

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)

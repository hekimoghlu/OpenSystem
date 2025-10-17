/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#include "RemoteMediaSessionCoordinator.h"

#if ENABLE(MEDIA_SESSION_COORDINATOR)

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "RemoteMediaSessionCoordinatorMessages.h"
#include "RemoteMediaSessionCoordinatorProxyMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <wtf/CompletionHandler.h>
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
extern WTFLogChannel LogMedia;
}

namespace WebKit {

using namespace PAL;
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaSessionCoordinator);

Ref<RemoteMediaSessionCoordinator> RemoteMediaSessionCoordinator::create(WebPage& page, const String& identifier)
{
    return adoptRef(*new RemoteMediaSessionCoordinator(page, identifier));
}

RemoteMediaSessionCoordinator::RemoteMediaSessionCoordinator(WebPage& page, const String& identifier)
    : m_page(page)
    , m_identifier(identifier)
{
    WebProcess::singleton().addMessageReceiver(Messages::RemoteMediaSessionCoordinator::messageReceiverName(), m_page->identifier(), *this);
}

RemoteMediaSessionCoordinator::~RemoteMediaSessionCoordinator()
{
    WebProcess::singleton().removeMessageReceiver(Messages::RemoteMediaSessionCoordinator::messageReceiverName(), m_page->identifier());
}

Ref<WebPage> RemoteMediaSessionCoordinator::protectedPage() const
{
    return m_page.get();
}

void RemoteMediaSessionCoordinator::join(CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    protectedPage()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinatorProxy::Join { }, [weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& exception) mutable {
        if (!weakThis) {
            callback(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            callback(exception->toException());
            return;
        }

        callback({ });
    });
}

void RemoteMediaSessionCoordinator::leave()
{
    protectedPage()->send(Messages::RemoteMediaSessionCoordinatorProxy::Leave { });
}

void RemoteMediaSessionCoordinator::seekTo(double time, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, time);
    protectedPage()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinatorProxy::CoordinateSeekTo { time }, [weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& exception) mutable {
        if (!weakThis) {
            callback(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            callback(exception->toException());
            return;
        }

        callback({ });
    });
}

void RemoteMediaSessionCoordinator::play(CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    protectedPage()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinatorProxy::CoordinatePlay { }, [weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& exception) mutable {
        if (!weakThis) {
            callback(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            callback(exception->toException());
            return;
        }

        callback({ });
    });
}

void RemoteMediaSessionCoordinator::pause(CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    protectedPage()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinatorProxy::CoordinatePause { }, [weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& exception) mutable {
        if (!weakThis) {
            callback(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            callback(exception->toException());
            return;
        }

        callback({ });
    });
}

void RemoteMediaSessionCoordinator::setTrack(const String& trackIdentifier, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    protectedPage()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinatorProxy::CoordinateSetTrack { trackIdentifier }, [weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& exception) mutable {
        if (!weakThis) {
            callback(Exception { ExceptionCode::InvalidStateError });
            return;
        }

        if (exception) {
            callback(exception->toException());
            return;
        }

        callback({ });
    });
}

void RemoteMediaSessionCoordinator::positionStateChanged(const std::optional<WebCore::MediaPositionState>& state)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    protectedPage()->send(Messages::RemoteMediaSessionCoordinatorProxy::PositionStateChanged { state });
}

void RemoteMediaSessionCoordinator::readyStateChanged(WebCore::MediaSessionReadyState state)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, state);
    protectedPage()->send(Messages::RemoteMediaSessionCoordinatorProxy::ReadyStateChanged { state });
}

void RemoteMediaSessionCoordinator::playbackStateChanged(WebCore::MediaSessionPlaybackState state)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, state);
    protectedPage()->send(Messages::RemoteMediaSessionCoordinatorProxy::PlaybackStateChanged { state });
}

void RemoteMediaSessionCoordinator::trackIdentifierChanged(const String& identifier)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, identifier);
    protectedPage()->send(Messages::RemoteMediaSessionCoordinatorProxy::TrackIdentifierChanged { identifier });
}

void RemoteMediaSessionCoordinator::seekSessionToTime(double time, CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, time);
    if (auto coordinatorClient = client())
        coordinatorClient->seekSessionToTime(time, WTFMove((completionHandler)));
    else
        completionHandler(false);
}

void RemoteMediaSessionCoordinator::playSession(std::optional<double> atTime, std::optional<MonotonicTime> hostTime, CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    if (auto coordinatorClient = client())
        coordinatorClient->playSession(WTFMove(atTime), WTFMove(hostTime), WTFMove((completionHandler)));
    else
        completionHandler(false);
}

void RemoteMediaSessionCoordinator::pauseSession(CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    if (auto coordinatorClient = client())
        coordinatorClient->pauseSession(WTFMove((completionHandler)));
    else
        completionHandler(false);
}

void RemoteMediaSessionCoordinator::setSessionTrack(const String& trackIdentifier, CompletionHandler<void(bool)>&& completionHandler)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, trackIdentifier);
    if (auto coordinatorClient = client())
        coordinatorClient->setSessionTrack(trackIdentifier, WTFMove((completionHandler)));
    else
        completionHandler(false);
}

void RemoteMediaSessionCoordinator::coordinatorStateChanged(WebCore::MediaSessionCoordinatorState state)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, state);
    if (auto coordinatorClient = client())
        coordinatorClient->coordinatorStateChanged(state);
}

WTFLogChannel& RemoteMediaSessionCoordinator::logChannel() const
{
    return JOIN_LOG_CHANNEL_WITH_PREFIX(LOG_CHANNEL_PREFIX, Media);
}


}

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)

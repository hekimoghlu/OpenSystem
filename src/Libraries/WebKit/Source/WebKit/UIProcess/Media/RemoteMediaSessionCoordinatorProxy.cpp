/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#include "RemoteMediaSessionCoordinatorProxy.h"

#if ENABLE(MEDIA_SESSION_COORDINATOR)

#include "Logging.h"
#include "MediaSessionCoordinatorProxyPrivate.h"
#include "RemoteMediaSessionCoordinatorMessages.h"
#include "RemoteMediaSessionCoordinatorProxyMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/ExceptionData.h>
#include <wtf/Logger.h>
#include <wtf/LoggerHelper.h>
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
extern WTFLogChannel LogMedia;
}

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaSessionCoordinatorProxy);

Ref<RemoteMediaSessionCoordinatorProxy> RemoteMediaSessionCoordinatorProxy::create(WebPageProxy& webPageProxy, Ref<MediaSessionCoordinatorProxyPrivate>&& privateCoordinator)
{
    return adoptRef(*new RemoteMediaSessionCoordinatorProxy(webPageProxy, WTFMove(privateCoordinator)));
}

RemoteMediaSessionCoordinatorProxy::RemoteMediaSessionCoordinatorProxy(WebPageProxy& webPageProxy, Ref<MediaSessionCoordinatorProxyPrivate>&& privateCoordinator)
    : m_webPageProxy(webPageProxy)
    , m_privateCoordinator(WTFMove(privateCoordinator))
#if !RELEASE_LOG_DISABLED
    , m_logger(webPageProxy.logger())
    , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
#endif
{
    protectedPrivateCoordinator()->setClient(*this);
    webPageProxy.protectedLegacyMainFrameProcess()->addMessageReceiver(Messages::RemoteMediaSessionCoordinatorProxy::messageReceiverName(), m_webPageProxy->webPageIDInMainFrameProcess(), *this);
}

RemoteMediaSessionCoordinatorProxy::~RemoteMediaSessionCoordinatorProxy()
{
    protectedWebPageProxy()->protectedLegacyMainFrameProcess()->removeMessageReceiver(Messages::RemoteMediaSessionCoordinatorProxy::messageReceiverName(), m_webPageProxy->webPageIDInMainFrameProcess());
}

std::optional<SharedPreferencesForWebProcess> RemoteMediaSessionCoordinatorProxy::sharedPreferencesForWebProcess() const
{
    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_webPageProxy->legacyMainFrameProcess().sharedPreferencesForWebProcess();
}

void RemoteMediaSessionCoordinatorProxy::join(MediaSessionCommandCompletionHandler&& completionHandler)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    protectedPrivateCoordinator()->join([this, protectedThis = Ref { *this }, identifier, completionHandler = WTFMove(completionHandler)] (std::optional<ExceptionData>&& exception) mutable {
        ALWAYS_LOG(identifier, "completion");
        completionHandler(WTFMove(exception));
    });
}

void RemoteMediaSessionCoordinatorProxy::leave()
{
    ALWAYS_LOG(LOGIDENTIFIER);

    protectedPrivateCoordinator()->leave();
}

void RemoteMediaSessionCoordinatorProxy::coordinateSeekTo(double time, MediaSessionCommandCompletionHandler&& completionHandler)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier, time);

    protectedPrivateCoordinator()->seekTo(time, [this, protectedThis = Ref { *this }, identifier, completionHandler = WTFMove(completionHandler)] (std::optional<ExceptionData>&& exception) mutable {
        ALWAYS_LOG(identifier, "completion");
        completionHandler(WTFMove(exception));
    });
}

void RemoteMediaSessionCoordinatorProxy::coordinatePlay(MediaSessionCommandCompletionHandler&& completionHandler)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    protectedPrivateCoordinator()->play([this, protectedThis = Ref { *this }, identifier, completionHandler = WTFMove(completionHandler)] (std::optional<ExceptionData>&& exception) mutable {
        ALWAYS_LOG(identifier, "completion");
        completionHandler(WTFMove(exception));
    });
}

void RemoteMediaSessionCoordinatorProxy::coordinatePause(MediaSessionCommandCompletionHandler&& completionHandler)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    protectedPrivateCoordinator()->pause([this, protectedThis = Ref { *this }, identifier, completionHandler = WTFMove(completionHandler)] (std::optional<ExceptionData>&& exception) mutable {
        ALWAYS_LOG(identifier, "completion");
        completionHandler(WTFMove(exception));
    });
}

void RemoteMediaSessionCoordinatorProxy::coordinateSetTrack(const String& track, MediaSessionCommandCompletionHandler&& completionHandler)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    protectedPrivateCoordinator()->setTrack(track, [this, protectedThis = Ref { *this }, identifier, completionHandler = WTFMove(completionHandler)] (std::optional<ExceptionData>&& exception) mutable {
        ALWAYS_LOG(identifier, "completion");
        completionHandler(WTFMove(exception));
    });
}

void RemoteMediaSessionCoordinatorProxy::positionStateChanged(const std::optional<WebCore::MediaPositionState>& state)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    protectedPrivateCoordinator()->positionStateChanged(state);
}

void RemoteMediaSessionCoordinatorProxy::playbackStateChanged(MediaSessionPlaybackState state)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    protectedPrivateCoordinator()->playbackStateChanged(state);
}

void RemoteMediaSessionCoordinatorProxy::readyStateChanged(MediaSessionReadyState state)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    protectedPrivateCoordinator()->readyStateChanged(state);
}

void RemoteMediaSessionCoordinatorProxy::trackIdentifierChanged(const String& identifier)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    protectedPrivateCoordinator()->trackIdentifierChanged(identifier);
}

void RemoteMediaSessionCoordinatorProxy::seekSessionToTime(double time, CompletionHandler<void(bool)>&& callback)
{
    protectedWebPageProxy()->protectedLegacyMainFrameProcess()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinator::SeekSessionToTime { time }, callback, m_webPageProxy->webPageIDInMainFrameProcess());
}

void RemoteMediaSessionCoordinatorProxy::playSession(std::optional<double> atTime, std::optional<MonotonicTime> hostTime, CompletionHandler<void(bool)>&& callback)
{
    protectedWebPageProxy()->protectedLegacyMainFrameProcess()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinator::PlaySession { WTFMove(atTime), WTFMove(hostTime) }, callback, m_webPageProxy->webPageIDInMainFrameProcess());
}

void RemoteMediaSessionCoordinatorProxy::pauseSession(CompletionHandler<void(bool)>&& callback)
{
    protectedWebPageProxy()->protectedLegacyMainFrameProcess()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinator::PauseSession { }, callback, m_webPageProxy->webPageIDInMainFrameProcess());
}

void RemoteMediaSessionCoordinatorProxy::setSessionTrack(const String& trackId, CompletionHandler<void(bool)>&& callback)
{
    protectedWebPageProxy()->protectedLegacyMainFrameProcess()->sendWithAsyncReply(Messages::RemoteMediaSessionCoordinator::SetSessionTrack { trackId }, callback, m_webPageProxy->webPageIDInMainFrameProcess());
}

void RemoteMediaSessionCoordinatorProxy::coordinatorStateChanged(WebCore::MediaSessionCoordinatorState state)
{
    protectedWebPageProxy()->protectedLegacyMainFrameProcess()->send(Messages::RemoteMediaSessionCoordinator::CoordinatorStateChanged { state }, m_webPageProxy->webPageIDInMainFrameProcess());
}

Ref<WebPageProxy> RemoteMediaSessionCoordinatorProxy::protectedWebPageProxy()
{
    return m_webPageProxy.get();
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& RemoteMediaSessionCoordinatorProxy::logChannel() const
{
    return JOIN_LOG_CHANNEL_WITH_PREFIX(LOG_CHANNEL_PREFIX, Media);
}
#endif

} // namespace WebKit

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)

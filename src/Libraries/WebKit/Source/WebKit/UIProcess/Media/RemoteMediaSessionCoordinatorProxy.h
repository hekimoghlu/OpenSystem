/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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

#include "MediaSessionCoordinatorProxyPrivate.h"
#include "MessageReceiver.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
struct ExceptionData;
}

namespace WebKit {

class WebPageProxy;
struct SharedPreferencesForWebProcess;

class RemoteMediaSessionCoordinatorProxy
    : private IPC::MessageReceiver
    , public RefCounted<RemoteMediaSessionCoordinatorProxy>
    , public WebCore::MediaSessionCoordinatorClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaSessionCoordinatorProxy);
public:
    static Ref<RemoteMediaSessionCoordinatorProxy> create(WebPageProxy&, Ref<MediaSessionCoordinatorProxyPrivate>&&);
    ~RemoteMediaSessionCoordinatorProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    void seekTo(double, CompletionHandler<void(bool)>&&);
    void play(CompletionHandler<void(bool)>&&);
    void pause(CompletionHandler<void(bool)>&&);
    void setTrack(const String&, CompletionHandler<void(bool)>&&);

    USING_CAN_MAKE_WEAKPTR(MediaSessionCoordinatorClient);

private:
    explicit RemoteMediaSessionCoordinatorProxy(WebPageProxy&, Ref<MediaSessionCoordinatorProxyPrivate>&&);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Receivers.
    void join(MediaSessionCommandCompletionHandler&&);
    void leave();
    void coordinateSeekTo(double, MediaSessionCommandCompletionHandler&&);
    void coordinatePlay(MediaSessionCommandCompletionHandler&&);
    void coordinatePause(MediaSessionCommandCompletionHandler&&);
    void coordinateSetTrack(const String&, MediaSessionCommandCompletionHandler&&);
    void positionStateChanged(const std::optional<WebCore::MediaPositionState>&);
    void readyStateChanged(WebCore::MediaSessionReadyState);
    void playbackStateChanged(WebCore::MediaSessionPlaybackState);
    void trackIdentifierChanged(const String&);

    // MediaSessionCoordinatorClient
    void seekSessionToTime(double, CompletionHandler<void(bool)>&&) final;
    void playSession(std::optional<double> atTime, std::optional<MonotonicTime> hostTime, CompletionHandler<void(bool)>&&) final;
    void pauseSession(CompletionHandler<void(bool)>&&) final;
    void setSessionTrack(const String&, CompletionHandler<void(bool)>&&) final;
    void coordinatorStateChanged(WebCore::MediaSessionCoordinatorState) final;

    Ref<MediaSessionCoordinatorProxyPrivate> protectedPrivateCoordinator() { return m_privateCoordinator; }
    Ref<WebPageProxy> protectedWebPageProxy();

#if !RELEASE_LOG_DISABLED
    const WTF::Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "RemoteMediaSessionCoordinatorProxy"_s; }
    WTFLogChannel& logChannel() const;
#endif

    WeakRef<WebPageProxy> m_webPageProxy;
    Ref<MediaSessionCoordinatorProxyPrivate> m_privateCoordinator;
#if !RELEASE_LOG_DISABLED
    Ref<const WTF::Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebKit

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)

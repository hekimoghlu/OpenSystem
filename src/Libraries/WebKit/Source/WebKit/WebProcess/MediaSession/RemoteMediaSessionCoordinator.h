/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#include "MessageReceiver.h"
#include <WebCore/MediaSessionCoordinatorPrivate.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace IPC {
class Connection;
class Decoder;
class MessageReceiver;
}

namespace WebKit {

class WebPage;

class RemoteMediaSessionCoordinator final : public WebCore::MediaSessionCoordinatorPrivate, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaSessionCoordinator);
public:
    static Ref<RemoteMediaSessionCoordinator> create(WebPage&, const String&);
    ~RemoteMediaSessionCoordinator();

    void ref() const final { WebCore::MediaSessionCoordinatorPrivate::ref(); }
    void deref() const final { WebCore::MediaSessionCoordinatorPrivate::deref(); }

private:
    explicit RemoteMediaSessionCoordinator(WebPage&, const String&);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // MessageReceivers.
    void seekSessionToTime(double, CompletionHandler<void(bool)>&&);
    void playSession(std::optional<double>, std::optional<MonotonicTime>, CompletionHandler<void(bool)>&&);
    void pauseSession(CompletionHandler<void(bool)>&&);
    void setSessionTrack(const String&, CompletionHandler<void(bool)>&&);
    void coordinatorStateChanged(WebCore::MediaSessionCoordinatorState);

    // MediaSessionCoordinatorPrivate overrides.
    String identifier() const final { return m_identifier; }
    void join(CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;
    void leave() final;
    void seekTo(double, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;
    void play(CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;
    void pause(CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;
    void setTrack(const String&, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;

    void positionStateChanged(const std::optional<WebCore::MediaPositionState>&) final;
    void readyStateChanged(WebCore::MediaSessionReadyState) final;
    void playbackStateChanged(WebCore::MediaSessionPlaybackState) final;
    void trackIdentifierChanged(const String&) final;

    ASCIILiteral logClassName() const { return "RemoteMediaSessionCoordinator"_s; }
    WTFLogChannel& logChannel() const;

    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_page;
    String m_identifier;
};

} // namespace WebKit

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)

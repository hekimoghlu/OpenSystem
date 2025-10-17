/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#if ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)

#include "MessageReceiver.h"
#include "RemoteMediaSourceIdentifier.h"
#include "RemoteSourceBufferProxy.h"
#include <WebCore/MediaSourcePrivate.h>
#include <WebCore/MediaSourcePrivateClient.h>
#include <wtf/MediaTime.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class ContentType;
class PlatformTimeRanges;
}

namespace WebKit {

class RemoteMediaPlayerManagerProxy;
class RemoteMediaPlayerProxy;

class RemoteMediaSourceProxy final
    : public WebCore::MediaSourcePrivateClient
    , private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaSourceProxy);
public:
    RemoteMediaSourceProxy(RemoteMediaPlayerManagerProxy&, RemoteMediaSourceIdentifier, RemoteMediaPlayerProxy&);
    virtual ~RemoteMediaSourceProxy();

    void ref() const final { WebCore::MediaSourcePrivateClient::ref(); }
    void deref() const final { WebCore::MediaSourcePrivateClient::deref(); }

    void setMediaPlayers(RemoteMediaPlayerProxy&, WebCore::MediaPlayerPrivateInterface*);

    // MediaSourcePrivateClient overrides
    void setPrivateAndOpen(Ref<WebCore::MediaSourcePrivate>&&) final;
    void reOpen() final;
    Ref<WebCore::MediaTimePromise> waitForTarget(const WebCore::SeekTarget&) final;
    Ref<WebCore::MediaPromise> seekToTime(const MediaTime&) final;
    RefPtr<WebCore::MediaSourcePrivate> mediaSourcePrivate() const final { return m_private; }

#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(uint64_t) final;
#endif

    void failedToCreateRenderer(RendererType) final;

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    using AddSourceBufferCallback = CompletionHandler<void(WebCore::MediaSourcePrivate::AddStatus, std::optional<RemoteSourceBufferIdentifier>)>;
    void addSourceBuffer(const WebCore::ContentType&, AddSourceBufferCallback&&);
    void durationChanged(const MediaTime&);
    void bufferedChanged(WebCore::PlatformTimeRanges&&);
    void markEndOfStream(WebCore::MediaSourcePrivate::EndOfStreamStatus);
    void unmarkEndOfStream();
    void setMediaPlayerReadyState(WebCore::MediaPlayerEnums::ReadyState);
    void setTimeFudgeFactor(const MediaTime&);
    void attached();
    void shutdown();

    void disconnect();
    RefPtr<GPUConnectionToWebProcess> connectionToWebProcess() const;

    WeakPtr<RemoteMediaPlayerManagerProxy> m_manager;
    RemoteMediaSourceIdentifier m_identifier;
    RefPtr<WebCore::MediaSourcePrivate> m_private;
    WeakPtr<RemoteMediaPlayerProxy> m_remoteMediaPlayerProxy;

    Vector<RefPtr<RemoteSourceBufferProxy>> m_sourceBuffers;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)

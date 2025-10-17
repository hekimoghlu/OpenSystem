/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MessageReceiver.h"
#include "SandboxExtension.h"
#include <WebCore/HTMLMediaElementIdentifier.h>
#include <WebCore/MediaPlayer.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/VideoTarget.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/Logger.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

#if ENABLE(LINEAR_MEDIA_PLAYER)
#include <WebCore/VideoReceiverEndpoint.h>
#endif
#if ENABLE(MEDIA_SOURCE)
#include "RemoteMediaSourceIdentifier.h"
#endif

namespace WebKit {

class RemoteMediaPlayerProxy;
struct RemoteMediaPlayerConfiguration;
struct RemoteMediaPlayerProxyConfiguration;
struct SharedPreferencesForWebProcess;
class RemoteMediaSourceProxy;
class VideoReceiverEndpointMessage;
class VideoReceiverSwapEndpointsMessage;

class RemoteMediaPlayerManagerProxy
    : public RefCounted<RemoteMediaPlayerManagerProxy>, public IPC::MessageReceiver
{
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaPlayerManagerProxy);
public:
    static Ref<RemoteMediaPlayerManagerProxy> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
    {
        return adoptRef(*new RemoteMediaPlayerManagerProxy(gpuConnectionToWebProcess));
    }

    ~RemoteMediaPlayerManagerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    RefPtr<GPUConnectionToWebProcess> gpuConnectionToWebProcess() { return m_gpuConnectionToWebProcess.get(); }
    void clear();

#if !RELEASE_LOG_DISABLED
    Logger& logger() { return m_logger; }
#endif

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }
    bool didReceiveSyncMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& encoder) { return didReceiveSyncMessage(connection, decoder, encoder); }
    void didReceivePlayerMessage(IPC::Connection&, IPC::Decoder&);
    bool didReceiveSyncPlayerMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    RefPtr<WebCore::MediaPlayer> mediaPlayer(std::optional<WebCore::MediaPlayerIdentifier>);

    std::optional<WebCore::ShareableBitmap::Handle> bitmapImageForCurrentTime(WebCore::MediaPlayerIdentifier);

#if ENABLE(LINEAR_MEDIA_PLAYER)
    WebCore::PlatformVideoTarget videoTargetForIdentifier(const std::optional<WebCore::VideoReceiverEndpointIdentifier>&);
    WebCore::PlatformVideoTarget takeVideoTargetForMediaElementIdentifier(WebCore::HTMLMediaElementIdentifier, WebCore::MediaPlayerIdentifier);
    void handleVideoReceiverEndpointMessage(const VideoReceiverEndpointMessage&);
    void handleVideoReceiverSwapEndpointsMessage(const VideoReceiverSwapEndpointsMessage&);
#endif

#if ENABLE(MEDIA_SOURCE)
    RefPtr<RemoteMediaSourceProxy> pendingMediaSource(RemoteMediaSourceIdentifier);
    void registerMediaSource(RemoteMediaSourceIdentifier, RemoteMediaSourceProxy&);
    void invalidateMediaSource(RemoteMediaSourceIdentifier);
#endif

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;
private:
    explicit RemoteMediaPlayerManagerProxy(GPUConnectionToWebProcess&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    void createMediaPlayer(WebCore::MediaPlayerIdentifier, WebCore::MediaPlayerClientIdentifier, WebCore::MediaPlayerEnums::MediaEngineIdentifier, RemoteMediaPlayerProxyConfiguration&&);
    void deleteMediaPlayer(WebCore::MediaPlayerIdentifier);

    // Media player factory
    void getSupportedTypes(WebCore::MediaPlayerEnums::MediaEngineIdentifier, CompletionHandler<void(Vector<String>&&)>&&);
    void supportsTypeAndCodecs(WebCore::MediaPlayerEnums::MediaEngineIdentifier, const WebCore::MediaEngineSupportParameters&&, CompletionHandler<void(WebCore::MediaPlayer::SupportsType)>&&);
    void supportsKeySystem(WebCore::MediaPlayerEnums::MediaEngineIdentifier, const String&&, const String&&, CompletionHandler<void(bool)>&&);

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const { return "RemoteMediaPlayerManagerProxy"; }
    WTFLogChannel& logChannel() const;
    uint64_t logIdentifier() const { return m_logIdentifier; }
#endif

    HashMap<WebCore::MediaPlayerIdentifier, Ref<RemoteMediaPlayerProxy>> m_proxies;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;

#if ENABLE(LINEAR_MEDIA_PLAYER)
    HashMap<WebCore::VideoReceiverEndpointIdentifier, WebCore::PlatformVideoTarget> m_videoTargetCache;
    struct VideoRecevierEndpointCacheEntry {
        Markable<WebCore::MediaPlayerIdentifier> playerIdentifier;
        Markable<WebCore::VideoReceiverEndpointIdentifier> endpointIdentifier;
    };
    HashMap<WebCore::HTMLMediaElementIdentifier, VideoRecevierEndpointCacheEntry> m_videoReceiverEndpointCache;
#endif

#if ENABLE(MEDIA_SOURCE)
    HashMap<RemoteMediaSourceIdentifier, RefPtr<RemoteMediaSourceProxy>> m_pendingMediaSources;
#endif

#if !RELEASE_LOG_DISABLED
    uint64_t m_logIdentifier { 0 };
    Ref<Logger> m_logger;
#endif
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

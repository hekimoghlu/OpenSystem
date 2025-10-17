/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

#include "GPUProcessConnection.h"
#include <WebCore/MediaPlayer.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class MediaPlayerPrivateInterface;
}

namespace WebKit {

class MediaPlayerPrivateRemote;
class RemoteMediaPlayerMIMETypeCache;
class WebProcess;
struct PlatformTextTrackData;
struct TrackPrivateRemoteConfiguration;
struct WebProcessCreationParameters;

class RemoteMediaPlayerManager
    : public GPUProcessConnection::Client
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteMediaPlayerManager> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaPlayerManager);
public:
    static Ref<RemoteMediaPlayerManager> create();
    ~RemoteMediaPlayerManager();

    void setUseGPUProcess(bool);

    GPUProcessConnection& gpuProcessConnection();
    Ref<GPUProcessConnection> protectedGPUProcessConnection();

    void didReceivePlayerMessage(IPC::Connection&, IPC::Decoder&);

    void deleteRemoteMediaPlayer(WebCore::MediaPlayerIdentifier);

    std::optional<WebCore::MediaPlayerIdentifier> findRemotePlayerId(const WebCore::MediaPlayerPrivateInterface*);

    RemoteMediaPlayerMIMETypeCache& typeCache(WebCore::MediaPlayerEnums::MediaEngineIdentifier);

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

    void initialize(const WebProcessCreationParameters&);

private:
    RemoteMediaPlayerManager();
    Ref<WebCore::MediaPlayerPrivateInterface> createRemoteMediaPlayer(WebCore::MediaPlayer*, WebCore::MediaPlayerEnums::MediaEngineIdentifier);

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    friend class MediaPlayerRemoteFactory;
    void getSupportedTypes(WebCore::MediaPlayerEnums::MediaEngineIdentifier, HashSet<String>&);
    WebCore::MediaPlayer::SupportsType supportsTypeAndCodecs(WebCore::MediaPlayerEnums::MediaEngineIdentifier, const WebCore::MediaEngineSupportParameters&);
    bool supportsKeySystem(WebCore::MediaPlayerEnums::MediaEngineIdentifier, const String& keySystem, const String& mimeType);

    HashMap<WebCore::MediaPlayerIdentifier, ThreadSafeWeakPtr<MediaPlayerPrivateRemote>> m_players;
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

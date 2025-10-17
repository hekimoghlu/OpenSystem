/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#include "RemoteMediaPlayerManager.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "MediaPlayerPrivateRemote.h"
#include "RemoteMediaPlayerConfiguration.h"
#include "RemoteMediaPlayerMIMETypeCache.h"
#include "RemoteMediaPlayerManagerProxyMessages.h"
#include "RemoteMediaPlayerProxyConfiguration.h"
#include "SampleBufferDisplayLayerManager.h"
#include "WebProcess.h"
#include "WebProcessCreationParameters.h"
#include <WebCore/ContentTypeUtilities.h>
#include <WebCore/MediaPlayer.h>
#include <wtf/HashFunctions.h>
#include <wtf/HashMap.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include <WebCore/MediaPlayerPrivateMediaStreamAVFObjC.h>
#endif

namespace WebKit {

using namespace PAL;
using namespace WebCore;

class MediaPlayerRemoteFactory final : public MediaPlayerFactory {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MediaPlayerRemoteFactory);
public:
    MediaPlayerRemoteFactory(MediaPlayerEnums::MediaEngineIdentifier remoteEngineIdentifier, RemoteMediaPlayerManager& manager)
        : m_remoteEngineIdentifier(remoteEngineIdentifier)
        , m_manager(manager)
    {
    }

    MediaPlayerEnums::MediaEngineIdentifier identifier() const final { return m_remoteEngineIdentifier; };

    Ref<MediaPlayerPrivateInterface> createMediaEnginePlayer(MediaPlayer* player) const final
    {
        return protectedManager()->createRemoteMediaPlayer(player, m_remoteEngineIdentifier);
    }

    void getSupportedTypes(HashSet<String>& types) const final
    {
        return protectedManager()->getSupportedTypes(m_remoteEngineIdentifier, types);
    }

    MediaPlayer::SupportsType supportsTypeAndCodecs(const MediaEngineSupportParameters& parameters) const final
    {
        return protectedManager()->supportsTypeAndCodecs(m_remoteEngineIdentifier, parameters);
    }

    HashSet<SecurityOriginData> originsInMediaCache(const String& path) const final
    {
        ASSERT_NOT_REACHED_WITH_MESSAGE("RemoteMediaPlayerManager does not support cache management");
        return { };
    }

    void clearMediaCache(const String& path, WallTime modifiedSince) const final
    {
        ASSERT_NOT_REACHED_WITH_MESSAGE("RemoteMediaPlayerManager does not support cache management");
    }

    void clearMediaCacheForOrigins(const String& path, const HashSet<SecurityOriginData>& origins) const final
    {
        ASSERT_NOT_REACHED_WITH_MESSAGE("RemoteMediaPlayerManager does not support cache management");
    }

    bool supportsKeySystem(const String& keySystem, const String& mimeType) const final
    {
        return protectedManager()->supportsKeySystem(m_remoteEngineIdentifier, keySystem, mimeType);
    }

private:
    Ref<RemoteMediaPlayerManager> protectedManager() const { return m_manager.get().releaseNonNull(); }

    MediaPlayerEnums::MediaEngineIdentifier m_remoteEngineIdentifier;
    ThreadSafeWeakPtr<RemoteMediaPlayerManager> m_manager; // Cannot be null.
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaPlayerManager);

Ref<RemoteMediaPlayerManager> RemoteMediaPlayerManager::create()
{
    return adoptRef(*new RemoteMediaPlayerManager);
}

RemoteMediaPlayerManager::RemoteMediaPlayerManager() = default;

RemoteMediaPlayerManager::~RemoteMediaPlayerManager() = default;

using RemotePlayerTypeCache = HashMap<MediaPlayerEnums::MediaEngineIdentifier, std::unique_ptr<RemoteMediaPlayerMIMETypeCache>, WTF::IntHash<MediaPlayerEnums::MediaEngineIdentifier>, WTF::StrongEnumHashTraits<MediaPlayerEnums::MediaEngineIdentifier>>;
static RemotePlayerTypeCache& mimeCaches()
{
    static NeverDestroyed<RemotePlayerTypeCache> caches;
    return caches;
}

RemoteMediaPlayerMIMETypeCache& RemoteMediaPlayerManager::typeCache(MediaPlayerEnums::MediaEngineIdentifier remoteEngineIdentifier)
{
    auto& cachePtr = mimeCaches().add(remoteEngineIdentifier, nullptr).iterator->value;
    if (!cachePtr)
        cachePtr = makeUnique<RemoteMediaPlayerMIMETypeCache>(*this, remoteEngineIdentifier);

    return *cachePtr;
}

void RemoteMediaPlayerManager::initialize(const WebProcessCreationParameters& parameters)
{
#if PLATFORM(COCOA)
    if (parameters.mediaMIMETypes.isEmpty())
        return;

    auto& cache = typeCache(MediaPlayerEnums::MediaEngineIdentifier::AVFoundation);
    if (cache.isEmpty())
        cache.addSupportedTypes(parameters.mediaMIMETypes);
#else
    UNUSED_PARAM(parameters);
#endif
}

Ref<MediaPlayerPrivateInterface> RemoteMediaPlayerManager::createRemoteMediaPlayer(MediaPlayer* player, MediaPlayerEnums::MediaEngineIdentifier remoteEngineIdentifier)
{
    RemoteMediaPlayerProxyConfiguration proxyConfiguration;
    proxyConfiguration.referrer = player->referrer();
    proxyConfiguration.userAgent = player->userAgent();
    proxyConfiguration.sourceApplicationIdentifier = player->sourceApplicationIdentifier();
#if PLATFORM(IOS_FAMILY)
    proxyConfiguration.networkInterfaceName = player->mediaPlayerNetworkInterfaceName();
#endif
    proxyConfiguration.audioOutputDeviceId = player->audioOutputDeviceId();
    proxyConfiguration.mediaContentTypesRequiringHardwareSupport = player->mediaContentTypesRequiringHardwareSupport();
    proxyConfiguration.renderingCanBeAccelerated = player->renderingCanBeAccelerated();
    proxyConfiguration.preferredAudioCharacteristics = player->preferredAudioCharacteristics();
#if !RELEASE_LOG_DISABLED
    proxyConfiguration.logIdentifier = reinterpret_cast<uint64_t>(player->mediaPlayerLogIdentifier());
#endif
    proxyConfiguration.shouldUsePersistentCache = player->shouldUsePersistentCache();
    proxyConfiguration.isVideo = player->isVideoPlayer();

#if PLATFORM(COCOA)
    proxyConfiguration.outOfBandTrackData = player->outOfBandTrackSources().map([](auto& track) {
        return track->data();
    });
#endif

    auto documentSecurityOrigin = player->documentSecurityOrigin();
    proxyConfiguration.documentSecurityOrigin = documentSecurityOrigin;

    proxyConfiguration.presentationSize = player->presentationSize();
    proxyConfiguration.videoLayerSize = player->videoLayerSize();

    proxyConfiguration.allowedMediaContainerTypes = player->allowedMediaContainerTypes();
    proxyConfiguration.allowedMediaCodecTypes = player->allowedMediaCodecTypes();
    proxyConfiguration.allowedMediaVideoCodecIDs = player->allowedMediaVideoCodecIDs();
    proxyConfiguration.allowedMediaAudioCodecIDs = player->allowedMediaAudioCodecIDs();
    proxyConfiguration.allowedMediaCaptionFormatTypes = player->allowedMediaCaptionFormatTypes();
    proxyConfiguration.playerContentBoxRect = player->playerContentBoxRect();

    proxyConfiguration.prefersSandboxedParsing = player->prefersSandboxedParsing();
#if PLATFORM(IOS_FAMILY)
    proxyConfiguration.canShowWhileLocked = player->canShowWhileLocked();
#endif

    auto identifier = MediaPlayerIdentifier::generate();
    auto clientIdentifier = player->clientIdentifier();
    gpuProcessConnection().connection().send(Messages::RemoteMediaPlayerManagerProxy::CreateMediaPlayer(identifier, clientIdentifier, remoteEngineIdentifier, proxyConfiguration), 0);

    auto remotePlayer = MediaPlayerPrivateRemote::create(player, remoteEngineIdentifier, identifier, *this);
    m_players.add(identifier, remotePlayer.get());

    return remotePlayer;
}

void RemoteMediaPlayerManager::deleteRemoteMediaPlayer(MediaPlayerIdentifier identifier)
{
    m_players.remove(identifier);
    gpuProcessConnection().connection().send(Messages::RemoteMediaPlayerManagerProxy::DeleteMediaPlayer(identifier), 0);
}

std::optional<MediaPlayerIdentifier> RemoteMediaPlayerManager::findRemotePlayerId(const MediaPlayerPrivateInterface* player)
{
    for (auto pair : m_players) {
        if (pair.value.get() == player)
            return pair.key;
    }

    return std::nullopt;
}

void RemoteMediaPlayerManager::getSupportedTypes(MediaPlayerEnums::MediaEngineIdentifier remoteEngineIdentifier, HashSet<String>& result)
{
    result = typeCache(remoteEngineIdentifier).supportedTypes();
}

MediaPlayer::SupportsType RemoteMediaPlayerManager::supportsTypeAndCodecs(MediaPlayerEnums::MediaEngineIdentifier remoteEngineIdentifier, const MediaEngineSupportParameters& parameters)
{
#if ENABLE(MEDIA_STREAM)
    if (parameters.isMediaStream)
        return MediaPlayer::SupportsType::IsNotSupported;
#endif

    if (!contentTypeMeetsContainerAndCodecTypeRequirements(parameters.type, parameters.allowedMediaContainerTypes, parameters.allowedMediaCodecTypes))
        return MediaPlayer::SupportsType::IsNotSupported;

    return typeCache(remoteEngineIdentifier).supportsTypeAndCodecs(parameters);
}

bool RemoteMediaPlayerManager::supportsKeySystem(MediaPlayerEnums::MediaEngineIdentifier, const String& keySystem, const String& mimeType)
{
    return false;
}

void RemoteMediaPlayerManager::didReceivePlayerMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (ObjectIdentifier<MediaPlayerIdentifierType>::isValidIdentifier(decoder.destinationID())) {
        if (RefPtr player = m_players.get(ObjectIdentifier<MediaPlayerIdentifierType>(decoder.destinationID())).get())
            player->didReceiveMessage(connection, decoder);
    }
}

void RemoteMediaPlayerManager::setUseGPUProcess(bool useGPUProcess)
{
    auto registerEngine = [this](MediaEngineRegistrar registrar, MediaPlayerEnums::MediaEngineIdentifier remoteEngineIdentifier) {
        registrar(makeUnique<MediaPlayerRemoteFactory>(remoteEngineIdentifier, *this));
    };

    RemoteMediaPlayerSupport::setRegisterRemotePlayerCallback(useGPUProcess ? WTFMove(registerEngine) : RemoteMediaPlayerSupport::RegisterRemotePlayerCallback());

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
    if (useGPUProcess) {
        WebCore::SampleBufferDisplayLayer::setCreator([](auto& client) -> RefPtr<WebCore::SampleBufferDisplayLayer> {
            return WebProcess::singleton().ensureGPUProcessConnection().sampleBufferDisplayLayerManager().createLayer(client);
        });
        WebCore::MediaPlayerPrivateMediaStreamAVFObjC::setNativeImageCreator([](auto& videoFrame) {
            return WebProcess::singleton().ensureGPUProcessConnection().videoFrameObjectHeapProxy().getNativeImage(videoFrame);
        });
    }
#endif
}

GPUProcessConnection& RemoteMediaPlayerManager::gpuProcessConnection()
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection) {
        gpuProcessConnection = &WebProcess::singleton().ensureGPUProcessConnection();
        m_gpuProcessConnection = gpuProcessConnection;
        gpuProcessConnection = &WebProcess::singleton().ensureGPUProcessConnection();
        gpuProcessConnection->addClient(*this);
    }

    return *gpuProcessConnection;
}

Ref<GPUProcessConnection> RemoteMediaPlayerManager::protectedGPUProcessConnection()
{
    return gpuProcessConnection();
}

void RemoteMediaPlayerManager::gpuProcessConnectionDidClose(GPUProcessConnection& connection)
{
    ASSERT(m_gpuProcessConnection.get() == &connection);

    m_gpuProcessConnection = nullptr;

    for (auto& player : copyToVector(m_players.values())) {
        if (RefPtr protectedPlayer = player.get())
            protectedPlayer->player()->reloadAndResumePlaybackIfNeeded();
        ASSERT_WITH_MESSAGE(!player.get(), "reloadAndResumePlaybackIfNeeded should destroy this player and construct a new one");
    }
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

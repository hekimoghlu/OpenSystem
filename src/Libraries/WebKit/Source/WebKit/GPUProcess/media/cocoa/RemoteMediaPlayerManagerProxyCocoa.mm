/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include "RemoteMediaPlayerManagerProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "VideoReceiverEndpointMessage.h"
#include <WebCore/MediaPlayerPrivate.h>
#include <WebCore/VideoTargetFactory.h>
#include <wtf/LoggerHelper.h>

namespace WebKit {

#if ENABLE(LINEAR_MEDIA_PLAYER)
PlatformVideoTarget RemoteMediaPlayerManagerProxy::videoTargetForIdentifier(const std::optional<WebCore::VideoReceiverEndpointIdentifier>& identifier)
{
    if (identifier)
        return m_videoTargetCache.get(*identifier);
    return nullptr;
}

PlatformVideoTarget RemoteMediaPlayerManagerProxy::takeVideoTargetForMediaElementIdentifier(WebCore::HTMLMediaElementIdentifier mediaElementIdentifier, WebCore::MediaPlayerIdentifier playerIdentifier)
{
    auto cachedEntry = m_videoReceiverEndpointCache.find(mediaElementIdentifier);
    if (cachedEntry == m_videoReceiverEndpointCache.end())
        return nullptr;

    if (cachedEntry->value.playerIdentifier != playerIdentifier) {
        ALWAYS_LOG(LOGIDENTIFIER, "moving target from player ", cachedEntry->value.playerIdentifier->loggingString(), " to player ", playerIdentifier.loggingString());
        if (RefPtr mediaPlayer = this->mediaPlayer(cachedEntry->value.playerIdentifier))
            mediaPlayer->setVideoTarget(nullptr);
        cachedEntry->value.playerIdentifier = playerIdentifier;
    }

    return videoTargetForIdentifier(cachedEntry->value.endpointIdentifier);
}

void RemoteMediaPlayerManagerProxy::handleVideoReceiverEndpointMessage(const VideoReceiverEndpointMessage& endpointMessage)
{
    // A message with an empty endpoint signals that the VideoTarget should be uncached and
    // removed from the existing player.
    if (!endpointMessage.endpoint()) {
        m_videoTargetCache.remove(endpointMessage.endpointIdentifier());
        auto cacheEntry = m_videoReceiverEndpointCache.takeOptional(endpointMessage.mediaElementIdentifier());
        if (!cacheEntry)
            return;

        if (RefPtr mediaPlayer = this->mediaPlayer(cacheEntry->playerIdentifier))
            mediaPlayer->setVideoTarget(nullptr);

        return;
    }

    // Handle caching or uncaching of VideoTargets. Because a VideoTarget can only be created
    // once during the lifetime of an endpoint, we should avoid re-creating these VideoTargets.
    auto ensureVideoTargetResult = m_videoTargetCache.ensure(endpointMessage.endpointIdentifier(), [&] {
        return WebCore::VideoTargetFactory::createTargetFromEndpoint(endpointMessage.endpoint());
    });
    PlatformVideoTarget cachedVideoTarget = ensureVideoTargetResult.iterator->value;

    auto cacheResult = m_videoReceiverEndpointCache.add(endpointMessage.mediaElementIdentifier(), VideoRecevierEndpointCacheEntry { endpointMessage.playerIdentifier(), endpointMessage.endpointIdentifier() });
    if (cacheResult.isNewEntry) {
        // If no entry for the specified mediaElementIdentifier exists, set the new target
        // on the specified player.
        if (RefPtr mediaPlayer = this->mediaPlayer(endpointMessage.playerIdentifier())) {
            ALWAYS_LOG(LOGIDENTIFIER, "New entry for player ", endpointMessage.playerIdentifier()->loggingString());
            mediaPlayer->setVideoTarget(cachedVideoTarget);
        }

        return;
    }

    // A previously cached entry already exists
    auto& cachedEntry = cacheResult.iterator->value;
    auto cachedPlayerIdentifier = cachedEntry.playerIdentifier;
    auto cachedEndpointIdentifier = cachedEntry.endpointIdentifier;

    // If nothing has actually changed, bail.
    if (cachedPlayerIdentifier == endpointMessage.playerIdentifier()
        && cachedEndpointIdentifier == endpointMessage.endpointIdentifier())
        return;

    // If the VideoTarget has been cleared, remove the entry from the cache entirely.
    if (!cachedVideoTarget) {
        if (RefPtr mediaPlayer = this->mediaPlayer(cachedPlayerIdentifier)) {
            ALWAYS_LOG(LOGIDENTIFIER, "Cache cleared; removing target from player ", cachedPlayerIdentifier->loggingString());
            mediaPlayer->setVideoTarget(nullptr);
        } else
            ALWAYS_LOG(LOGIDENTIFIER, "Cache cleared; no current player target");

        m_videoReceiverEndpointCache.remove(cacheResult.iterator);
        return;
    }

    RefPtr cachedPlayer = mediaPlayer(cachedPlayerIdentifier);

    if (cachedPlayerIdentifier != endpointMessage.playerIdentifier() && cachedPlayer) {
        // A endpoint can only be used by one MediaPlayer at a time, so if the playerIdentifier
        // has changed, first remove the endpoint from that cached MediaPlayer.
        ALWAYS_LOG(LOGIDENTIFIER, "Update entry; removing target from player ", cachedPlayerIdentifier->loggingString());
        cachedPlayer->setVideoTarget(nullptr);
    }

    // Then set the new target, which may have changed, on the specified MediaPlayer.
    if (RefPtr mediaPlayer = this->mediaPlayer(endpointMessage.playerIdentifier())) {
        ALWAYS_LOG(LOGIDENTIFIER, "Update entry; ", !cachedVideoTarget ? "removing target" : "setting target", " on player ", endpointMessage.playerIdentifier()->loggingString());
        mediaPlayer->setVideoTarget(cachedVideoTarget);
    }

    // Otherwise, update the cache entry with updated values.
    cachedEntry.playerIdentifier = endpointMessage.playerIdentifier();
    cachedEntry.endpointIdentifier = endpointMessage.endpointIdentifier();
}

void RemoteMediaPlayerManagerProxy::handleVideoReceiverSwapEndpointsMessage(const VideoReceiverSwapEndpointsMessage& swapMessage)
{
    auto sourceCacheEntry = m_videoReceiverEndpointCache.takeOptional(swapMessage.sourceMediaElementIdentifier());
    RefPtr sourcePlayer = mediaPlayer(swapMessage.sourceMediaPlayerIdentifier());
    auto sourceTarget = sourceCacheEntry ? videoTargetForIdentifier(sourceCacheEntry->endpointIdentifier) : nullptr;

    auto destinationCacheEntry = m_videoReceiverEndpointCache.takeOptional(swapMessage.destinationMediaElementIdentifier());
    RefPtr destinationPlayer = mediaPlayer(swapMessage.destinationMediaPlayerIdentifier());
    auto destinationTarget = destinationCacheEntry ? videoTargetForIdentifier(destinationCacheEntry->endpointIdentifier) : nullptr;

    ALWAYS_LOG(LOGIDENTIFIER, "swapping from media element ", swapMessage.sourceMediaElementIdentifier().loggingString(), " to media element ", swapMessage.destinationMediaElementIdentifier().loggingString());

    // To avoid two media players using the VideoTarget simultaneously, set both players
    // to have null targets before continuing
    if (sourcePlayer)
        sourcePlayer->setVideoTarget(nullptr);

    if (destinationPlayer)
        destinationPlayer->setVideoTarget(nullptr);

    if (sourcePlayer)
        sourcePlayer->setVideoTarget(destinationTarget);

    if (destinationPlayer)
        destinationPlayer->setVideoTarget(sourceTarget);

    if (sourceCacheEntry) {
        sourceCacheEntry->playerIdentifier = swapMessage.destinationMediaPlayerIdentifier();
        m_videoReceiverEndpointCache.set(swapMessage.destinationMediaElementIdentifier(), *sourceCacheEntry);
    }

    if (destinationCacheEntry) {
        destinationCacheEntry->playerIdentifier = swapMessage.sourceMediaPlayerIdentifier();
        m_videoReceiverEndpointCache.set(swapMessage.sourceMediaElementIdentifier(), *destinationCacheEntry);
    }
}

#endif

}

#endif

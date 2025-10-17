/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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

#if USE(EXTERNAL_HOLEPUNCH)

#include "MediaPlayerPrivate.h"
#include "PlatformLayer.h"
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CoordinatedPlatformLayerBufferProxy;

class MediaPlayerPrivateHolePunch
    : public MediaPlayerPrivateInterface
    , public CanMakeWeakPtr<MediaPlayerPrivateHolePunch>
    , public RefCounted<MediaPlayerPrivateHolePunch>
{
    WTF_MAKE_TZONE_ALLOCATED(MediaPlayerPrivateHolePunch);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    MediaPlayerPrivateHolePunch(MediaPlayer*);
    ~MediaPlayerPrivateHolePunch();

    constexpr MediaPlayerType mediaPlayerType() const final { return MediaPlayerType::HolePunch; }

    static void registerMediaEngine(MediaEngineRegistrar);

    void load(const String&) final;
#if ENABLE(MEDIA_SOURCE)
    void load(const URL&, const ContentType&, MediaSourcePrivateClient&) final { };
#endif
#if ENABLE(MEDIA_STREAM)
    void load(MediaStreamPrivate&) final { };
#endif
    void cancelLoad() final { };

    void play() final { };
    void pause() final { };

#if USE(COORDINATED_GRAPHICS)
    PlatformLayer* platformLayer() const final;
#endif

    FloatSize naturalSize() const final;

    bool hasVideo() const final { return false; };
    bool hasAudio() const final { return false; };

    void setPageIsVisible(bool) final { };

    bool seeking() const final { return false; }
    void seekToTarget(const SeekTarget&) final { }

    bool paused() const final { return false; };

    MediaPlayer::NetworkState networkState() const final { return m_networkState; };
    MediaPlayer::ReadyState readyState() const final { return MediaPlayer::ReadyState::HaveMetadata; };

    const PlatformTimeRanges& buffered() const final { return PlatformTimeRanges::emptyRanges(); };

    bool didLoadingProgress() const final { return false; };

    void setPresentationSize(const IntSize& size) final { m_size = size; };

    void paint(GraphicsContext&, const FloatRect&) final { };

    DestinationColorSpace colorSpace() final { return DestinationColorSpace::SRGB(); }

    bool supportsAcceleratedRendering() const final { return true; }

    bool shouldIgnoreIntrinsicSize() final { return true; }

    void pushNextHolePunchBuffer();
    void setNetworkState(MediaPlayer::NetworkState);

    static void getSupportedTypes(HashSet<String>&);

private:
    friend class MediaPlayerFactoryHolePunch;
    static MediaPlayer::SupportsType supportsType(const MediaEngineSupportParameters&);

    void notifyReadyState();

    ThreadSafeWeakPtr<MediaPlayer> m_player;
    IntSize m_size;
    RunLoop::Timer m_readyTimer;
    MediaPlayer::NetworkState m_networkState;
#if USE(COORDINATED_GRAPHICS)
    RefPtr<CoordinatedPlatformLayerBufferProxy> m_contentsBufferProxy;
#endif

};
}
#endif // USE(EXTERNAL_HOLEPUNCH)

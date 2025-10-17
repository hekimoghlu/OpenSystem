/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#include "MediaPlayerPrivateHolePunch.h"

#if USE(EXTERNAL_HOLEPUNCH)
#include "CoordinatedPlatformLayerBufferHolePunch.h"
#include "CoordinatedPlatformLayerBufferProxy.h"
#include "MediaPlayer.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaPlayerPrivateHolePunch);

static const FloatSize s_holePunchDefaultFrameSize(1280, 720);

MediaPlayerPrivateHolePunch::MediaPlayerPrivateHolePunch(MediaPlayer* player)
    : m_player(player)
    , m_readyTimer(RunLoop::main(), this, &MediaPlayerPrivateHolePunch::notifyReadyState)
    , m_networkState(MediaPlayer::NetworkState::Empty)
#if USE(COORDINATED_GRAPHICS)
    , m_contentsBufferProxy(CoordinatedPlatformLayerBufferProxy::create())
#endif
{
    pushNextHolePunchBuffer();

    // Delay the configuration of the HTMLMediaElement, as during this stage this is not
    // the MediaPlayer private yet and calls from HTMLMediaElement won't reach this.
    m_readyTimer.startOneShot(0_s);
}

MediaPlayerPrivateHolePunch::~MediaPlayerPrivateHolePunch()
{
}

#if USE(COORDINATED_GRAPHICS)
PlatformLayer* MediaPlayerPrivateHolePunch::platformLayer() const
{
    return m_contentsBufferProxy.get();
}
#endif

FloatSize MediaPlayerPrivateHolePunch::naturalSize() const
{
    // When using the holepuch we may not be able to get the video frames size, so we can't use
    // it. But we need to report some non empty naturalSize for the player's GraphicsLayer
    // to be properly created.
    return s_holePunchDefaultFrameSize;
}

void MediaPlayerPrivateHolePunch::pushNextHolePunchBuffer()
{
    m_contentsBufferProxy->setDisplayBuffer(CoordinatedPlatformLayerBufferHolePunch::create(m_size));
}

static HashSet<String>& mimeTypeCache()
{
    static NeverDestroyed<HashSet<String>> cache;
    static bool typeListInitialized = false;

    if (typeListInitialized)
        return cache;

    const ASCIILiteral mimeTypes[] = {
        "video/holepunch"_s
    };

    for (unsigned i = 0; i < (sizeof(mimeTypes) / sizeof(*mimeTypes)); ++i)
        cache.get().add(mimeTypes[i]);

    typeListInitialized = true;

    return cache;
}

void MediaPlayerPrivateHolePunch::getSupportedTypes(HashSet<String>& types)
{
    types = mimeTypeCache();
}

MediaPlayer::SupportsType MediaPlayerPrivateHolePunch::supportsType(const MediaEngineSupportParameters& parameters)
{
    auto containerType = parameters.type.containerType();

    // Spec says we should not return "probably" if the codecs string is empty.
    if (!containerType.isEmpty() && mimeTypeCache().contains(containerType)) {
        if (parameters.type.codecs().isEmpty())
            return MediaPlayer::SupportsType::MayBeSupported;

        return MediaPlayer::SupportsType::IsSupported;
    }

    return MediaPlayer::SupportsType::IsNotSupported;
}

class MediaPlayerFactoryHolePunch final : public MediaPlayerFactory {
private:
    MediaPlayerEnums::MediaEngineIdentifier identifier() const final { return MediaPlayerEnums::MediaEngineIdentifier::HolePunch; };

    Ref<MediaPlayerPrivateInterface> createMediaEnginePlayer(MediaPlayer* player) const final
    {
        return adoptRef(*new MediaPlayerPrivateHolePunch(player));
    }

    void getSupportedTypes(HashSet<String>& types) const final
    {
        return MediaPlayerPrivateHolePunch::getSupportedTypes(types);
    }

    MediaPlayer::SupportsType supportsTypeAndCodecs(const MediaEngineSupportParameters& parameters) const final
    {
        return MediaPlayerPrivateHolePunch::supportsType(parameters);
    }
};

void MediaPlayerPrivateHolePunch::registerMediaEngine(MediaEngineRegistrar registrar)
{
    registrar(makeUnique<MediaPlayerFactoryHolePunch>());
}

void MediaPlayerPrivateHolePunch::notifyReadyState()
{
    // Notify the ready state so the GraphicsLayer gets created.
    if (auto player = m_player.get())
        player->readyStateChanged();
}

void MediaPlayerPrivateHolePunch::setNetworkState(MediaPlayer::NetworkState networkState)
{
    m_networkState = networkState;
    if (auto player = m_player.get())
        player->networkStateChanged();
}

void MediaPlayerPrivateHolePunch::load(const String&)
{
    auto player = m_player.get();
    if (!player)
        return;

    auto mimeType = player->contentMIMEType();
    if (mimeType.isEmpty() || !mimeTypeCache().contains(mimeType))
        setNetworkState(MediaPlayer::NetworkState::FormatError);
}

} // namespace WebCore
#endif // USE(EXTERNAL_HOLEPUNCH)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#import "config.h"
#import "MediaPlayerPrivateRemote.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA)

#import "RemoteAudioSourceProvider.h"
#import "RemoteMediaPlayerProxyMessages.h"
#import "VideoLayerRemote.h"
#import <WebCore/ColorSpaceCG.h>
#import <WebCore/VideoLayerManager.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/MachSendRight.h>

#import <WebCore/CoreVideoSoftLink.h>

namespace WebKit {
using namespace WebCore;

#if ENABLE(VIDEO_PRESENTATION_MODE)
PlatformLayerContainer MediaPlayerPrivateRemote::createVideoFullscreenLayer()
{
    return adoptNS([[CALayer alloc] init]);
}
#endif

void MediaPlayerPrivateRemote::pushVideoFrameMetadata(WebCore::VideoFrameMetadata&& videoFrameMetadata, RemoteVideoFrameProxy::Properties&& properties)
{
    auto videoFrame = RemoteVideoFrameProxy::create(protectedConnection(), videoFrameObjectHeapProxy(), WTFMove(properties));
    if (!m_isGatheringVideoFrameMetadata)
        return;
    m_videoFrameMetadata = WTFMove(videoFrameMetadata);
    m_videoFrameGatheredWithVideoFrameMetadata = WTFMove(videoFrame);
}

RefPtr<NativeImage> MediaPlayerPrivateRemote::nativeImageForCurrentTime()
{
    if (readyState() < MediaPlayer::ReadyState::HaveCurrentData)
        return { };

    RefPtr videoFrame = videoFrameForCurrentTime();
    if (!videoFrame)
        return nullptr;

    return WebProcess::singleton().ensureGPUProcessConnection().videoFrameObjectHeapProxy().getNativeImage(*videoFrame);
}

WebCore::DestinationColorSpace MediaPlayerPrivateRemote::colorSpace()
{
    if (readyState() < MediaPlayer::ReadyState::HaveCurrentData)
        return DestinationColorSpace::SRGB();

    auto sendResult = connection().sendSync(Messages::RemoteMediaPlayerProxy::ColorSpace(), m_id);
    auto [colorSpace] = sendResult.takeReplyOr(DestinationColorSpace::SRGB());
    return colorSpace;
}

void MediaPlayerPrivateRemote::layerHostingContextIdChanged(std::optional<WebKit::LayerHostingContextID>&& inlineLayerHostingContextId, const FloatSize& presentationSize)
{
    RefPtr player = m_player.get();
    if (!player)
        return;

    if (!inlineLayerHostingContextId) {
        m_videoLayer = nullptr;
        m_videoLayerManager->didDestroyVideoLayer();
        return;
    }
    setLayerHostingContextID(inlineLayerHostingContextId.value());
    player->videoLayerSizeDidChange(presentationSize);
}

WebCore::FloatSize MediaPlayerPrivateRemote::videoLayerSize() const
{
    if (RefPtr player = m_player.get())
        return player->videoLayerSize();
    return { };
}

void MediaPlayerPrivateRemote::setVideoLayerSizeFenced(const FloatSize& size, WTF::MachSendRight&& machSendRight)
{
    connection().send(Messages::RemoteMediaPlayerProxy::SetVideoLayerSizeFenced(size, WTFMove(machSendRight)), m_id);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && PLATFORM(COCOA)

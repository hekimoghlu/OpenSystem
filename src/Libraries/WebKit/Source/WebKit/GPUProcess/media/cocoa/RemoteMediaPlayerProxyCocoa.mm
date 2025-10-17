/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#import "RemoteMediaPlayerProxy.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA)

#import "LayerHostingContext.h"
#import "MediaPlayerPrivateRemoteMessages.h"
#import <QuartzCore/QuartzCore.h>
#import <WebCore/FloatSize.h>
#import <WebCore/IOSurface.h>
#import <WebCore/VideoFrameCV.h>
#import <wtf/MachSendRight.h>

#if USE(EXTENSIONKIT)
#import <BrowserEngineKit/BELayerHierarchy.h>
#import <BrowserEngineKit/BELayerHierarchyHandle.h>
#import <BrowserEngineKit/BELayerHierarchyHostingTransactionCoordinator.h>
#endif

namespace WebKit {

void RemoteMediaPlayerProxy::setVideoLayerSizeIfPossible(const WebCore::FloatSize& size)
{
    if (!m_inlineLayerHostingContext || !m_inlineLayerHostingContext->rootLayer() || size.isEmpty())
        return;

    ALWAYS_LOG(LOGIDENTIFIER, size.width(), "x", size.height());

    // We do not want animations here.
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [m_inlineLayerHostingContext->rootLayer() setFrame:CGRectMake(0, 0, size.width(), size.height())];
    [CATransaction commit];
}

void RemoteMediaPlayerProxy::mediaPlayerFirstVideoFrameAvailable()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    setVideoLayerSizeIfPossible(m_configuration.videoLayerSize);
    protectedConnection()->send(Messages::MediaPlayerPrivateRemote::FirstVideoFrameAvailable(), m_id);
}

void RemoteMediaPlayerProxy::mediaPlayerRenderingModeChanged()
{
    ALWAYS_LOG(LOGIDENTIFIER);

    auto* layer = protectedPlayer()->platformLayer();
    if (layer && !m_inlineLayerHostingContext) {
        LayerHostingContextOptions contextOptions;
#if USE(EXTENSIONKIT)
        contextOptions.useHostable = true;
#endif
#if PLATFORM(IOS_FAMILY)
        contextOptions.canShowWhileLocked = m_configuration.canShowWhileLocked;
#endif
        m_inlineLayerHostingContext = LayerHostingContext::createForExternalHostingProcess(contextOptions);
        if (m_configuration.videoLayerSize.isEmpty())
            m_configuration.videoLayerSize = enclosingIntRect(FloatRect(layer.frame)).size();
        auto& size = m_configuration.videoLayerSize;
        [layer setFrame:CGRectMake(0, 0, size.width(), size.height())];
        protectedConnection()->send(Messages::MediaPlayerPrivateRemote::LayerHostingContextIdChanged(m_inlineLayerHostingContext->contextID(), size), m_id);
        for (auto& request : std::exchange(m_layerHostingContextIDRequests, { }))
            request(m_inlineLayerHostingContext->contextID());
    } else if (!layer && m_inlineLayerHostingContext) {
        m_inlineLayerHostingContext = nullptr;
        protectedConnection()->send(Messages::MediaPlayerPrivateRemote::LayerHostingContextIdChanged(std::nullopt, { }), m_id);
    }

    if (m_inlineLayerHostingContext)
        m_inlineLayerHostingContext->setRootLayer(layer);

    protectedConnection()->send(Messages::MediaPlayerPrivateRemote::RenderingModeChanged(), m_id);
}

void RemoteMediaPlayerProxy::requestHostingContextID(CompletionHandler<void(LayerHostingContextID)>&& completionHandler)
{
    if (m_inlineLayerHostingContext) {
        completionHandler(m_inlineLayerHostingContext->contextID());
        return;
    }

    m_layerHostingContextIDRequests.append(WTFMove(completionHandler));
}

void RemoteMediaPlayerProxy::setVideoLayerSizeFenced(const WebCore::FloatSize& size, WTF::MachSendRight&& machSendRight)
{
    ALWAYS_LOG(LOGIDENTIFIER, size.width(), "x", size.height());

#if USE(EXTENSIONKIT)
    RetainPtr<BELayerHierarchyHostingTransactionCoordinator> hostingUpdateCoordinator;
#endif

    if (m_inlineLayerHostingContext) {
#if USE(EXTENSIONKIT)
        hostingUpdateCoordinator = LayerHostingContext::createHostingUpdateCoordinator(machSendRight.sendRight());
        [hostingUpdateCoordinator addLayerHierarchy:m_inlineLayerHostingContext->hostable().get()];
#else
        m_inlineLayerHostingContext->setFencePort(machSendRight.sendRight());
#endif
    }

    m_configuration.videoLayerSize = size;
    setVideoLayerSizeIfPossible(size);

    protectedPlayer()->setVideoLayerSizeFenced(size, WTFMove(machSendRight));
#if USE(EXTENSIONKIT)
    [hostingUpdateCoordinator commit];
#endif
}

void RemoteMediaPlayerProxy::mediaPlayerOnNewVideoFrameMetadata(VideoFrameMetadata&& metadata, RetainPtr<CVPixelBufferRef>&& buffer)
{
    auto properties = protectedVideoFrameObjectHeap()->add(WebCore::VideoFrameCV::create({ }, false, VideoFrame::Rotation::None, WTFMove(buffer)));
    protectedConnection()->send(Messages::MediaPlayerPrivateRemote::PushVideoFrameMetadata(metadata, properties), m_id);
}

void RemoteMediaPlayerProxy::nativeImageForCurrentTime(CompletionHandler<void(std::optional<WTF::MachSendRight>&&, WebCore::DestinationColorSpace)>&& completionHandler)
{
    RefPtr player = m_player;
    if (!player) {
        completionHandler(std::nullopt, DestinationColorSpace::SRGB());
        return;
    }

    auto nativeImage = player->nativeImageForCurrentTime();
    if (!nativeImage) {
        completionHandler(std::nullopt, DestinationColorSpace::SRGB());
        return;
    }

    auto platformImage = nativeImage->platformImage();
    if (!platformImage) {
        completionHandler(std::nullopt, DestinationColorSpace::SRGB());
        return;
    }

    auto surface = WebCore::IOSurface::createFromImage(nullptr, platformImage.get());
    if (!surface) {
        completionHandler(std::nullopt, DestinationColorSpace::SRGB());
        return;
    }

    completionHandler(surface->createSendRight(), nativeImage->colorSpace());
}

void RemoteMediaPlayerProxy::colorSpace(CompletionHandler<void(WebCore::DestinationColorSpace)>&& completionHandler)
{
    RefPtr player = m_player;
    if (!player) {
        completionHandler(DestinationColorSpace::SRGB());
        return;
    }

    completionHandler(player->colorSpace());
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && PLATFORM(COCOA)

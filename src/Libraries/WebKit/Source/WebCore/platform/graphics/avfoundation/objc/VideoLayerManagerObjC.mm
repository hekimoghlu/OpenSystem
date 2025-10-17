/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#import "VideoLayerManagerObjC.h"

#import "Color.h"
#import "Logging.h"
#import "TextTrackRepresentation.h"
#import "WebCoreCALayerExtras.h"
#import "WebVideoContainerLayer.h"
#import <mach/mach_init.h>
#import <mach/mach_port.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/Logger.h>
#import <wtf/MachSendRight.h>
#import <wtf/TZoneMallocInlines.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

OBJC_CLASS AVPlayerLayer;

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoLayerManagerObjC);

#if !RELEASE_LOG_DISABLED
VideoLayerManagerObjC::VideoLayerManagerObjC(const Logger& logger, uint64_t logIdentifier)
    : m_logger(logger)
    , m_logIdentifier(logIdentifier)
{
}
#endif

VideoLayerManagerObjC::~VideoLayerManagerObjC()
{
}

PlatformLayer* VideoLayerManagerObjC::videoInlineLayer() const
{
    return m_videoInlineLayer.get();
}

void VideoLayerManagerObjC::setVideoLayer(PlatformLayer *videoLayer, FloatSize contentSize)
{
    ALWAYS_LOG(LOGIDENTIFIER, contentSize);

    m_videoLayer = videoLayer;
    [m_videoLayer web_disableAllActions];

    m_videoInlineLayer = adoptNS([[WebVideoContainerLayer alloc] init]);
    [m_videoInlineLayer setName:@"WebVideoContainerLayer"];
    [m_videoInlineLayer setFrame:CGRectMake(0, 0, contentSize.width(), contentSize.height())];
    [m_videoInlineLayer setContentsGravity:kCAGravityResizeAspect];
    if (PAL::isAVFoundationFrameworkAvailable() && [videoLayer isKindOfClass:PAL::getAVPlayerLayerClass()])
        [m_videoInlineLayer setPlayerLayer:(AVPlayerLayer *)videoLayer];

#if ENABLE(VIDEO_PRESENTATION_MODE)
    if (m_videoFullscreenLayer) {
        [m_videoLayer setFrame:m_videoFullscreenFrame];
        [m_videoFullscreenLayer insertSublayer:m_videoLayer.get() atIndex:0];
    } else
#endif
    {
        [m_videoLayer setFrame:m_videoInlineLayer.get().bounds];
        [m_videoInlineLayer insertSublayer:m_videoLayer.get() atIndex:0];
    }
}

void VideoLayerManagerObjC::didDestroyVideoLayer()
{
    ALWAYS_LOG(LOGIDENTIFIER);

#if ENABLE(VIDEO_PRESENTATION_MODE)
    setTextTrackRepresentationLayer(nil);
#endif
    [m_videoLayer removeFromSuperlayer];

    m_videoInlineLayer = nil;
    m_videoLayer = nil;
}

#if ENABLE(VIDEO_PRESENTATION_MODE)

PlatformLayer* VideoLayerManagerObjC::videoFullscreenLayer() const
{
    return m_videoFullscreenLayer.get();
}

void VideoLayerManagerObjC::setVideoFullscreenLayer(PlatformLayer *videoFullscreenLayer, WTF::Function<void()>&& completionHandler, PlatformImagePtr currentImage)
{
    if (m_videoFullscreenLayer == videoFullscreenLayer) {
        completionHandler();
        return;
    }

    ALWAYS_LOG(LOGIDENTIFIER);

    m_videoFullscreenLayer = videoFullscreenLayer;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];

    if (m_videoLayer) {
        CAContext *oldContext = [m_videoLayer context];

        if (m_videoInlineLayer && currentImage)
            [m_videoInlineLayer setContents:(__bridge id)currentImage.get()];

        if (m_videoFullscreenLayer) {
            [m_videoLayer setFrame:m_videoFullscreenFrame];
            [m_videoFullscreenLayer insertSublayer:m_videoLayer.get() atIndex:0];
        } else if (m_videoInlineLayer) {
            [m_videoLayer setFrame:[m_videoInlineLayer bounds]];
            [m_videoInlineLayer insertSublayer:m_videoLayer.get() atIndex:0];
        } else
            [m_videoLayer removeFromSuperlayer];

        CAContext *newContext = [m_videoLayer context];
        if (oldContext && newContext && oldContext != newContext) {
#if PLATFORM(MAC)
            oldContext.commitPriority = 0;
            newContext.commitPriority = 1;
#endif
            auto fencePort = MachSendRight::adopt([oldContext createFencePort]);
            [newContext setFencePort:fencePort.sendRight()];
        }
    }

    [CATransaction setCompletionBlock:makeBlockPtr([completionHandler = WTFMove(completionHandler)] {
        completionHandler();
    }).get()];

    [CATransaction commit];
}

FloatRect VideoLayerManagerObjC::videoFullscreenFrame() const
{
    return m_videoFullscreenFrame;
}

void VideoLayerManagerObjC::setVideoFullscreenFrame(FloatRect videoFullscreenFrame)
{
    ALWAYS_LOG(LOGIDENTIFIER, videoFullscreenFrame.x(), ", ", videoFullscreenFrame.y(), ", ", videoFullscreenFrame.width(), ", ", videoFullscreenFrame.height());

    m_videoFullscreenFrame = videoFullscreenFrame;
    if (!m_videoFullscreenLayer)
        return;

    [m_videoLayer setFrame:m_videoFullscreenFrame];
    syncTextTrackBounds();
}

void VideoLayerManagerObjC::updateVideoFullscreenInlineImage(PlatformImagePtr image)
{
    if (m_videoInlineLayer)
        [m_videoInlineLayer setContents:(__bridge id)image.get()];
}

#endif

void VideoLayerManagerObjC::syncTextTrackBounds()
{
#if ENABLE(VIDEO_PRESENTATION_MODE)
    if (!m_videoFullscreenLayer || !m_textTrackRepresentationLayer)
        return;

    if (m_textTrackRepresentationLayer.get().bounds == m_videoFullscreenFrame)
        return;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];

    [m_textTrackRepresentationLayer setFrame:m_videoFullscreenFrame];

    [CATransaction commit];
#endif
}

void VideoLayerManagerObjC::setTextTrackRepresentationLayer(PlatformLayer* representationLayer)
{
#if !ENABLE(VIDEO_PRESENTATION_MODE)
    UNUSED_PARAM(representationLayer);
#else
    ALWAYS_LOG(LOGIDENTIFIER);

    if (representationLayer == m_textTrackRepresentationLayer) {
        syncTextTrackBounds();
        return;
    }

    [CATransaction begin];
    [CATransaction setDisableActions:YES];

    if (m_textTrackRepresentationLayer)
        [m_textTrackRepresentationLayer removeFromSuperlayer];

    m_textTrackRepresentationLayer = representationLayer;

    if (m_videoFullscreenLayer && m_textTrackRepresentationLayer) {
        syncTextTrackBounds();
        [m_videoFullscreenLayer addSublayer:m_textTrackRepresentationLayer.get()];
    }

    [CATransaction commit];
#endif
}

WTFLogChannel& VideoLayerManagerObjC::logChannel() const
{
    return LogMedia;
}

}

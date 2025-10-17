/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#import "VideoLayerRemoteCocoa.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA)

#import "LayerHostingContext.h"
#import "MediaPlayerPrivateRemote.h"
#import "VideoLayerRemote.h"
#import <WebCore/FloatRect.h>
#import <WebCore/GeometryUtilities.h>
#import <WebCore/Timer.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/MachSendRight.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/WeakPtr.h>

// We want to wait for a short time after the completion of the animation (we choose 100 ms here) to fire the timer
// to avoid excessive XPC messages from the Web process to the GPU process.
static const Seconds PostAnimationDelay { 100_ms };

@implementation WKVideoLayerRemote {
    ThreadSafeWeakPtr<WebKit::MediaPlayerPrivateRemote> _mediaPlayerPrivateRemote;
    RetainPtr<CAContext> _context;
    WebCore::MediaPlayerEnums::VideoGravity _videoGravity;

    std::unique_ptr<WebCore::Timer> _resolveBoundsTimer;
    bool _shouldRestartWhenTimerFires;
    Seconds _delay;
}

- (instancetype)init
{
    self = [super init];
    if (!self)
        return nil;

    self.masksToBounds = YES;
    _resolveBoundsTimer = makeUnique<WebCore::Timer>([weakSelf = WeakObjCPtr<WKVideoLayerRemote>(self)] {
        auto localSelf = weakSelf.get();
        if (!localSelf)
            return;

        [localSelf resolveBounds];
    });
    _shouldRestartWhenTimerFires = false;

    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread(WKVideoLayerRemote.class, self))
        return;

    [super dealloc];
}

- (WebKit::MediaPlayerPrivateRemote*)mediaPlayerPrivateRemote
{
    return _mediaPlayerPrivateRemote.get().get();
}

- (void)setMediaPlayerPrivateRemote:(WebKit::MediaPlayerPrivateRemote*)mediaPlayerPrivateRemote
{
    _mediaPlayerPrivateRemote = *mediaPlayerPrivateRemote;
}

- (WebCore::MediaPlayerEnums::VideoGravity)videoGravity
{
    return _videoGravity;
}

- (void)setVideoGravity:(WebCore::MediaPlayerEnums::VideoGravity)videoGravity
{
    _videoGravity = videoGravity;
}

- (bool)resizePreservingGravity
{
    RefPtr<WebKit::MediaPlayerPrivateRemote> player = self.mediaPlayerPrivateRemote;
    if (player && player->inVideoFullscreenOrPictureInPicture())
        return true;
    
    return _videoGravity != WebCore::MediaPlayer::VideoGravity::Resize;
}

- (void)layoutSublayers
{
    auto* sublayers = [self sublayers];
    
    if ([sublayers count] != 1) {
        ASSERT_NOT_REACHED();
        return;
    }

    WebCore::FloatRect sourceVideoFrame = self.videoLayerFrame;
    WebCore::FloatRect targetVideoFrame = self.bounds;

    if (sourceVideoFrame == targetVideoFrame && CGAffineTransformIsIdentity(self.affineTransform))
        return;

    if (sourceVideoFrame.isEmpty()) {
        // The initial resize will have an empty videoLayerFrame, which makes
        // the subsequent calculations incorrect. When this happens, just do
        // the synchronous resize step instead.
        [self resolveBounds];
        return;
    }

    CGAffineTransform transform = CGAffineTransformIdentity;
    if ([self resizePreservingGravity]) {
        WebCore::FloatSize naturalSize { };
        if (RefPtr mediaPlayer = _mediaPlayerPrivateRemote.get())
            naturalSize = mediaPlayer->naturalSize();

        if (!naturalSize.isEmpty()) {
            // The video content will be sized within the remote layer, preserving aspect
            // ratio according to its naturalSize(), so use that natural size to determine
            // the scaling factor.
            auto naturalAspectRatio = naturalSize.aspectRatio();

            sourceVideoFrame = largestRectWithAspectRatioInsideRect(naturalAspectRatio, sourceVideoFrame);
            targetVideoFrame = largestRectWithAspectRatioInsideRect(naturalAspectRatio, targetVideoFrame);
        }
        auto scale = std::fmax(targetVideoFrame.width() / sourceVideoFrame.width(), targetVideoFrame.height() / sourceVideoFrame.height());
        transform = CGAffineTransformMakeScale(scale, scale);
    } else
        transform = CGAffineTransformMakeScale(targetVideoFrame.width() / sourceVideoFrame.width(), targetVideoFrame.height() / sourceVideoFrame.height());

    auto* videoSublayer = [sublayers objectAtIndex:0];
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [videoSublayer setPosition:CGPointMake(CGRectGetMidX(self.bounds), CGRectGetMidY(self.bounds))];
    [videoSublayer setAffineTransform:transform];
    [CATransaction commit];

    _context = [CAContext currentContext];
    NSTimeInterval animationDuration = [CATransaction animationDuration];

    _delay = Seconds(animationDuration) + PostAnimationDelay;
    if (_resolveBoundsTimer->isActive()) {
        _shouldRestartWhenTimerFires = true;
        _delay -= _resolveBoundsTimer->nextFireInterval();
        return;
    }
    _resolveBoundsTimer->startOneShot(_delay);
}

- (void)resolveBounds
{
    if (_shouldRestartWhenTimerFires) {
        _shouldRestartWhenTimerFires = false;
        _resolveBoundsTimer->startOneShot(_delay);
        return;
    }

    auto* sublayers = [self sublayers];
    if ([sublayers count] != 1) {
        ASSERT_NOT_REACHED();
        return;
    }

    auto* videoSublayer = [sublayers objectAtIndex:0];
    if (!CGRectIsEmpty(self.videoLayerFrame) && CGRectEqualToRect(self.videoLayerFrame, videoSublayer.bounds) && CGAffineTransformIsIdentity(videoSublayer.affineTransform))
        return;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];

    if (!CGRectEqualToRect(self.videoLayerFrame, self.bounds)) {
        self.videoLayerFrame = self.bounds;
        if (RefPtr<WebKit::MediaPlayerPrivateRemote> mediaPlayerPrivateRemote = self.mediaPlayerPrivateRemote) {
            MachSendRight fenceSendRight = MachSendRight::adopt([_context createFencePort]);
            mediaPlayerPrivateRemote->setVideoLayerSizeFenced(WebCore::FloatSize(self.videoLayerFrame.size), WTFMove(fenceSendRight));
        }
    }

    [videoSublayer setAffineTransform:CGAffineTransformIdentity];
    [videoSublayer setFrame:self.bounds];

    [CATransaction commit];
}

@end

namespace WebKit {

PlatformLayerContainer createVideoLayerRemote(MediaPlayerPrivateRemote* mediaPlayerPrivateRemote, LayerHostingContextID contextId, WebCore::MediaPlayerEnums::VideoGravity videoGravity, IntSize contentSize)
{
    // Initially, all the layers will be empty (both width and height are 0) and invisible.
    // The renderer will change the sizes of WKVideoLayerRemote to trigger layout of sublayers and make them visible.
    auto videoLayerRemote = adoptNS([[WKVideoLayerRemote alloc] init]);
    [videoLayerRemote setName:@"WKVideoLayerRemote"];
    [videoLayerRemote setVideoGravity:videoGravity];
    [videoLayerRemote setMediaPlayerPrivateRemote:mediaPlayerPrivateRemote];
    auto layerForHostContext = LayerHostingContext::createPlatformLayerForHostingContext(contextId).get();
    auto frame = CGRectMake(0, 0, contentSize.width(), contentSize.height());
    [videoLayerRemote setVideoLayerFrame:frame];
    [layerForHostContext setFrame:frame];
    [videoLayerRemote addSublayer:WTFMove(layerForHostContext)];

    return videoLayerRemote;
}

} // namespace WebKit

#endif

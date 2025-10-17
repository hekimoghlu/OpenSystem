/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#import "VideoPresentationInterfaceLMK.h"

#if ENABLE(LINEAR_MEDIA_PLAYER)

#import "LinearMediaKitExtras.h"
#import "LinearMediaKitSPI.h"
#import "PlaybackSessionInterfaceLMK.h"
#import "WKSLinearMediaPlayer.h"
#import "WKSLinearMediaTypes.h"
#import <QuartzCore/CALayer.h>
#import <UIKit/UIKit.h>
#import <WebCore/AudioSession.h>
#import <WebCore/Color.h>
#import <WebCore/IntRect.h>
#import <WebCore/WebAVPlayerLayerView.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/UUID.h>
#import <wtf/text/MakeString.h>

#import "WebKitSwiftSoftLink.h"

@interface WKLinearMediaKitCaptionsLayer : CALayer {
    ThreadSafeWeakPtr<WebKit::VideoPresentationInterfaceLMK> _parent;
}
- (id)initWithParent:(WebKit::VideoPresentationInterfaceLMK&)parent;
@end

@implementation WKLinearMediaKitCaptionsLayer
- (id)initWithParent:(WebKit::VideoPresentationInterfaceLMK&)parent
{
    self = [super init];
    if (!self)
        return nil;

    _parent = parent;
    return self;
}

- (void)layoutSublayers
{
    [super layoutSublayers];
    if (RefPtr parent = _parent.get())
        parent->captionsLayerBoundsChanged(self.bounds);
}
@end

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoPresentationInterfaceLMK);

VideoPresentationInterfaceLMK::~VideoPresentationInterfaceLMK()
{
}

Ref<VideoPresentationInterfaceLMK> VideoPresentationInterfaceLMK::create(WebCore::PlaybackSessionInterfaceIOS& playbackSessionInterface)
{
    return adoptRef(*new VideoPresentationInterfaceLMK(playbackSessionInterface));
}

VideoPresentationInterfaceLMK::VideoPresentationInterfaceLMK(WebCore::PlaybackSessionInterfaceIOS& playbackSessionInterface)
    : VideoPresentationInterfaceIOS { playbackSessionInterface }
{
}

WKSLinearMediaPlayer *VideoPresentationInterfaceLMK::linearMediaPlayer() const
{
    return playbackSessionInterface().linearMediaPlayer();
}

void VideoPresentationInterfaceLMK::setSpatialImmersive(bool immersive)
{
    linearMediaPlayer().spatialImmersive = immersive;
}

void VideoPresentationInterfaceLMK::setupFullscreen(const WebCore::FloatRect& initialRect, const WebCore::FloatSize& videoDimensions, UIView* parentView, WebCore::HTMLMediaElementEnums::VideoFullscreenMode mode, bool allowsPictureInPicturePlayback, bool standby, bool blocksReturnToFullscreenFromPictureInPicture)
{
    linearMediaPlayer().contentDimensions = videoDimensions;
    if (!linearMediaPlayer().enteredFromInline && playerViewController()) {
        playableViewController().wks_automaticallyDockOnFullScreenPresentation = NO;
        playableViewController().wks_dismissFullScreenOnExitingDocking = NO;
    }
    VideoPresentationInterfaceIOS::setupFullscreen(initialRect, videoDimensions, parentView, mode, allowsPictureInPicturePlayback, standby, blocksReturnToFullscreenFromPictureInPicture);
}

void VideoPresentationInterfaceLMK::finalizeSetup()
{
    RunLoop::main().dispatch([protectedThis = Ref { *this }] {
        if (RefPtr model = protectedThis->videoPresentationModel())
            model->didSetupFullscreen();
    });
}

void VideoPresentationInterfaceLMK::setupPlayerViewController()
{
    linearMediaPlayer().captionLayer = captionsLayer();
    linearMediaPlayer().contentType = WKSLinearMediaContentTypePlanar;

    ensurePlayableViewController();
}

void VideoPresentationInterfaceLMK::invalidatePlayerViewController()
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    m_playerViewController = nil;
}

void VideoPresentationInterfaceLMK::presentFullscreen(bool animated, Function<void(BOOL, NSError *)>&& completionHandler)
{
    playbackSessionInterface().startObservingNowPlayingMetadata();
    [linearMediaPlayer() enterFullscreenWithCompletionHandler:makeBlockPtr([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (BOOL success, NSError *error) {
        if (auto* playbackSessionModel = this->playbackSessionModel()) {
            playbackSessionModel->setSpatialTrackingLabel(m_spatialTrackingLabel);
            playbackSessionModel->setSoundStageSize(WebCore::AudioSessionSoundStageSize::Large);
        }
        completionHandler(success, error);
    }).get()];
}

void VideoPresentationInterfaceLMK::dismissFullscreen(bool animated, Function<void(BOOL, NSError *)>&& completionHandler)
{
    playbackSessionInterface().stopObservingNowPlayingMetadata();
    [linearMediaPlayer() exitFullscreenWithCompletionHandler:makeBlockPtr([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (BOOL success, NSError *error) {
        if (auto* playbackSessionModel = this->playbackSessionModel()) {
            playbackSessionModel->setSpatialTrackingLabel(nullString());
            playbackSessionModel->setSoundStageSize(WebCore::AudioSessionSoundStageSize::Automatic);
        }
        completionHandler(success, error);
    }).get()];
}

UIViewController *VideoPresentationInterfaceLMK::playerViewController() const
{
    return m_playerViewController.get();
}

void VideoPresentationInterfaceLMK::setContentDimensions(const WebCore::FloatSize& contentDimensions)
{
    linearMediaPlayer().contentDimensions = contentDimensions;
}

void VideoPresentationInterfaceLMK::setShowsPlaybackControls(bool showsPlaybackControls)
{
    linearMediaPlayer().showsPlaybackControls = showsPlaybackControls;
}

CALayer *VideoPresentationInterfaceLMK::captionsLayer()
{
    if (m_captionsLayer)
        return m_captionsLayer.get();

    m_captionsLayer = adoptNS([[WKLinearMediaKitCaptionsLayer alloc] initWithParent:*this]);
    [m_captionsLayer setName:@"Captions Layer"];

#if HAVE(SPATIAL_TRACKING_LABEL)
    m_spatialTrackingLayer = adoptNS([[CALayer alloc] init]);
    [m_spatialTrackingLayer setSeparatedState:kCALayerSeparatedStateTracked];
    m_spatialTrackingLabel = makeString("VideoPresentationInterfaceLMK Label: "_s, createVersion4UUIDString());
    [m_spatialTrackingLayer setValue:(NSString *)m_spatialTrackingLabel forKeyPath:@"separatedOptions.STSLabel"];
    [m_captionsLayer addSublayer:m_spatialTrackingLayer.get()];
#endif

    return m_captionsLayer.get();
}

void VideoPresentationInterfaceLMK::captionsLayerBoundsChanged(const WebCore::FloatRect& bounds)
{
#if HAVE(SPATIAL_TRACKING_LABEL)
    [m_spatialTrackingLayer setPosition:bounds.center()];
#endif
    if (RefPtr model = videoPresentationModel())
        model->setVideoFullscreenFrame(enclosingIntRect(bounds));
}

void VideoPresentationInterfaceLMK::setupCaptionsLayer(CALayer *, const WebCore::FloatSize& initialSize)
{
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [captionsLayer() removeFromSuperlayer];
    [captionsLayer() setAnchorPoint:CGPointZero];
    [captionsLayer() setBounds:CGRectMake(0, 0, initialSize.width(), initialSize.height())];
    [CATransaction commit];
}

LMPlayableViewController *VideoPresentationInterfaceLMK::playableViewController()
{
    ensurePlayableViewController();
    return m_playerViewController.get();
}

void VideoPresentationInterfaceLMK::ensurePlayableViewController()
{
    if (m_playerViewController)
        return;

    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    m_playerViewController = [linearMediaPlayer() makeViewController];
    [m_playerViewController view].alpha = 0;
}

void VideoPresentationInterfaceLMK::swapFullscreenModesWith(VideoPresentationInterfaceIOS& otherInterfaceIOS)
{
    auto& otherInterface = static_cast<VideoPresentationInterfaceLMK&>(otherInterfaceIOS);
    std::swap(m_playerViewController, otherInterface.m_playerViewController);

    auto currentMode = mode();
    auto previousMode = otherInterface.mode();

    setMode(previousMode, true);
    otherInterface.setMode(currentMode, true);
}

} // namespace WebKit

#endif // ENABLE(LINEAR_MEDIA_PLAYER)

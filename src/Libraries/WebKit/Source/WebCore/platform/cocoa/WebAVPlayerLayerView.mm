/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#import "WebAVPlayerLayerView.h"

#if PLATFORM(IOS_FAMILY) && HAVE(AVKIT)

#import "WebAVPlayerLayer.h"
#import <objc/message.h>
#import <objc/runtime.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>
#import <pal/ios/UIKitSoftLink.h>

#if HAVE(PICTUREINPICTUREPLAYERLAYERVIEW)
static NSString * const pictureInPicturePlayerLayerViewKey = @"_pictureInPicturePlayerLayerView";
#endif

SOFTLINK_AVKIT_FRAMEWORK()
SOFT_LINK_CLASS_OPTIONAL(AVKit, __AVPlayerLayerView)

namespace WebCore {

static Class WebAVPlayerLayerView_layerClass(id, SEL)
{
    return [WebAVPlayerLayer class];
}

static void WebAVPlayerLayerView_transferVideoViewTo(id aSelf, SEL, WebAVPlayerLayerView *targetPlayerLayerView)
{
    WebAVPlayerLayerView *playerLayerView = aSelf;
    RetainPtr videoView = [playerLayerView videoView];
    if (!videoView)
        return;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [videoView removeFromSuperview];
    [playerLayerView setVideoView:nil];
    [targetPlayerLayerView setVideoView:videoView.get()];
    [targetPlayerLayerView addSubview:videoView.get()];
    [CATransaction commit];
}

static AVPlayerController *WebAVPlayerLayerView_playerController(id aSelf, SEL)
{
    __AVPlayerLayerView *playerLayer = aSelf;
    WebAVPlayerLayer *webAVPlayerLayer = (WebAVPlayerLayer *)[playerLayer playerLayer];
    return [webAVPlayerLayer playerController];
}

static void WebAVPlayerLayerView_setPlayerController(id aSelf, SEL, AVPlayerController *playerController)
{
    __AVPlayerLayerView *playerLayerView = aSelf;
    WebAVPlayerLayer *webAVPlayerLayer = (WebAVPlayerLayer *)[playerLayerView playerLayer];
    [webAVPlayerLayer setPlayerController:playerController];
}

static AVPlayerLayer *WebAVPlayerLayerView_playerLayer(id aSelf, SEL)
{
    __AVPlayerLayerView *playerLayerView = aSelf;

    if ([get__AVPlayerLayerViewClass() instancesRespondToSelector:@selector(playerLayer)]) {
        objc_super superClass { playerLayerView, get__AVPlayerLayerViewClass() };
        auto superClassMethod = reinterpret_cast<AVPlayerLayer *(*)(objc_super *, SEL)>(objc_msgSendSuper);
        return superClassMethod(&superClass, @selector(playerLayer));
    }

    return (AVPlayerLayer *)[playerLayerView layer];
}

static UIView *WebAVPlayerLayerView_videoView(id aSelf, SEL)
{
    __AVPlayerLayerView *playerLayerView = aSelf;
    WebAVPlayerLayer *webAVPlayerLayer = (WebAVPlayerLayer *)[playerLayerView playerLayer];
    CALayer* videoLayer = [webAVPlayerLayer videoSublayer];
    if (!videoLayer || !videoLayer.delegate)
        return nil;
    ASSERT([[videoLayer delegate] isKindOfClass:PAL::getUIViewClass()]);
    return (UIView *)[videoLayer delegate];
}

static void WebAVPlayerLayerView_setVideoView(id aSelf, SEL, UIView *videoView)
{
    __AVPlayerLayerView *playerLayerView = aSelf;
    WebAVPlayerLayer *webAVPlayerLayer = (WebAVPlayerLayer *)[playerLayerView playerLayer];
    [webAVPlayerLayer setVideoSublayer:[videoView layer]];
}

#if HAVE(PICTUREINPICTUREPLAYERLAYERVIEW)
static void WebAVPlayerLayerView_startRoutingVideoToPictureInPicturePlayerLayerView(id aSelf, SEL)
{
    WebAVPlayerLayerView *playerLayerView = aSelf;
    auto *pipView = (WebAVPictureInPicturePlayerLayerView *)[playerLayerView pictureInPicturePlayerLayerView];

    auto *playerLayer = (WebAVPlayerLayer *)[playerLayerView playerLayer];
    auto *pipPlayerLayer = (WebAVPlayerLayer *)[pipView layer];
    [playerLayer setVideoGravity:AVLayerVideoGravityResizeAspectFill];
    [pipPlayerLayer setPresentationModel:playerLayer.presentationModel];
    [pipPlayerLayer setVideoSublayer:playerLayer.videoSublayer];
    [pipPlayerLayer setVideoDimensions:playerLayer.videoDimensions];
    [pipPlayerLayer setVideoGravity:playerLayer.videoGravity];
    [pipPlayerLayer setPlayerController:playerLayer.playerController];
    [pipView addSubview:playerLayerView.videoView];
    [pipPlayerLayer setCaptionsLayer:playerLayer.captionsLayer];
    [pipPlayerLayer layoutSublayers];
}

static void WebAVPlayerLayerView_stopRoutingVideoToPictureInPicturePlayerLayerView(id aSelf, SEL)
{
    WebAVPlayerLayerView *playerLayerView = aSelf;
    auto *pipView = (WebAVPictureInPicturePlayerLayerView *)[playerLayerView pictureInPicturePlayerLayerView];

    auto *playerLayer = (WebAVPlayerLayer *)[playerLayerView playerLayer];
    auto *pipPlayerLayer = (WebAVPlayerLayer *)[pipView layer];
    [playerLayerView addSubview:playerLayerView.videoView];
    [playerLayer setCaptionsLayer:pipPlayerLayer.captionsLayer];
    [playerLayer layoutSublayers];
}

static WebAVPictureInPicturePlayerLayerView *WebAVPlayerLayerView_pictureInPicturePlayerLayerView(id aSelf, SEL)
{
    WebAVPlayerLayerView *playerLayerView = aSelf;
    if (WebAVPictureInPicturePlayerLayerView *pipView = [playerLayerView valueForKey:pictureInPicturePlayerLayerViewKey])
        return pipView;

    auto pipView = adoptNS([allocWebAVPictureInPicturePlayerLayerViewInstance() initWithFrame:CGRectZero]);
    [playerLayerView setValue:pipView.get() forKey:pictureInPicturePlayerLayerViewKey];
    return pipView.get();
}
#endif // HAVE(PICTUREINPICTUREPLAYERLAYERVIEW)

static void WebAVPlayerLayerView_dealloc(id aSelf, SEL)
{
    WebAVPlayerLayerView *playerLayerView = aSelf;

#if HAVE(PICTUREINPICTUREPLAYERLAYERVIEW)
    [playerLayerView setValue:nil forKey:pictureInPicturePlayerLayerViewKey];
#endif
    objc_super superClass { playerLayerView, get__AVPlayerLayerViewClass() };
    auto super_dealloc = reinterpret_cast<void(*)(objc_super*, SEL)>(objc_msgSendSuper);
    super_dealloc(&superClass, @selector(dealloc));
}

#pragma mark - Methods

WebAVPlayerLayerView *allocWebAVPlayerLayerViewInstance()
{
    static Class theClass = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        ASSERT(get__AVPlayerLayerViewClass());
        theClass = objc_allocateClassPair(get__AVPlayerLayerViewClass(), "WebAVPlayerLayerView", 0);
        class_addMethod(theClass, @selector(dealloc), (IMP)WebAVPlayerLayerView_dealloc, "v@:");
        class_addMethod(theClass, @selector(transferVideoViewTo:), (IMP)WebAVPlayerLayerView_transferVideoViewTo, "v@:@");
        class_addMethod(theClass, @selector(setPlayerController:), (IMP)WebAVPlayerLayerView_setPlayerController, "v@:@");
        class_addMethod(theClass, @selector(playerController), (IMP)WebAVPlayerLayerView_playerController, "@@:");
        class_addMethod(theClass, @selector(setVideoView:), (IMP)WebAVPlayerLayerView_setVideoView, "v@:@");
        class_addMethod(theClass, @selector(videoView), (IMP)WebAVPlayerLayerView_videoView, "@@:");
        class_addMethod(theClass, @selector(playerLayer), (IMP)WebAVPlayerLayerView_playerLayer, "@@:");
#if HAVE(PICTUREINPICTUREPLAYERLAYERVIEW)
        class_addMethod(theClass, @selector(startRoutingVideoToPictureInPicturePlayerLayerView), (IMP)WebAVPlayerLayerView_startRoutingVideoToPictureInPicturePlayerLayerView, "v@:");
        class_addMethod(theClass, @selector(stopRoutingVideoToPictureInPicturePlayerLayerView), (IMP)WebAVPlayerLayerView_stopRoutingVideoToPictureInPicturePlayerLayerView, "v@:");
        class_addMethod(theClass, @selector(pictureInPicturePlayerLayerView), (IMP)WebAVPlayerLayerView_pictureInPicturePlayerLayerView, "@@:");
        
        class_addIvar(theClass, "_pictureInPicturePlayerLayerView", sizeof(WebAVPictureInPicturePlayerLayerView *), log2(sizeof(WebAVPictureInPicturePlayerLayerView *)), "@");
#endif

        objc_registerClassPair(theClass);
        Class metaClass = objc_getMetaClass("WebAVPlayerLayerView");
        class_addMethod(metaClass, @selector(layerClass), (IMP)WebAVPlayerLayerView_layerClass, "@@:");
    });
    return (WebAVPlayerLayerView *)[theClass alloc];
}

#if HAVE(PICTUREINPICTUREPLAYERLAYERVIEW)
static Class WebAVPictureInPicturePlayerLayerView_layerClass(id, SEL)
{
    return [WebAVPlayerLayer class];
}

WebAVPictureInPicturePlayerLayerView *allocWebAVPictureInPicturePlayerLayerViewInstance()
{
    static Class theClass = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        theClass = objc_allocateClassPair(PAL::getUIViewClass(), "WebAVPictureInPicturePlayerLayerView", 0);
        objc_registerClassPair(theClass);
        Class metaClass = objc_getMetaClass("WebAVPictureInPicturePlayerLayerView");
        class_addMethod(metaClass, @selector(layerClass), (IMP)WebAVPictureInPicturePlayerLayerView_layerClass, "@@:");
    });

    return (WebAVPictureInPicturePlayerLayerView *)[theClass alloc];
}
#endif

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY) && HAVE(AVKIT)

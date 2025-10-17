/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#import "WKAirPlayRoutePicker.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(AIRPLAY_PICKER)

#import "UIKitSPI.h"
#import "UIKitUtilities.h"
#import <WebCore/AudioSession.h>
#import <pal/spi/ios/MediaPlayerSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/SoftLinking.h>

ALLOW_DEPRECATED_DECLARATIONS_BEGIN

SOFT_LINK_FRAMEWORK(MediaPlayer)
SOFT_LINK_CLASS(MediaPlayer, MPAVRoutingController)
SOFT_LINK_CLASS(MediaPlayer, MPMediaControlsConfiguration)
SOFT_LINK_CLASS(MediaPlayer, MPMediaControlsViewController)

@interface MPMediaControlsConfiguration (WKMPMediaControlsConfiguration)
@property (nonatomic) BOOL sortByIsVideoRoute;
@end

enum {
    WKAirPlayRoutePickerRouteSharingPolicyDefault = 0,
    WKAirPlayRoutePickerRouteSharingPolicyLongFormAudio = 1,
    WKAirPlayRoutePickerRouteSharingPolicyIndependent = 2,
    WKAirPlayRoutePickerRouteSharingPolicyLongFormVideo = 3,
};
typedef NSInteger WKAirPlayRoutePickerRouteSharingPolicy;

@interface MPMediaControlsViewController (WKMPMediaControlsViewControllerPrivate)
- (instancetype)initWithConfiguration:(MPMediaControlsConfiguration *)configuration;
- (void)setOverrideRouteSharingPolicy:(WKAirPlayRoutePickerRouteSharingPolicy)routeSharingPolicy routingContextUID:(NSString *)routingContextUID;
@end

@implementation WKAirPlayRoutePicker {
    RetainPtr<MPMediaControlsViewController> _actionSheet;
}

- (void)dealloc
{
    [_actionSheet dismissViewControllerAnimated:0 completion:nil];
    [super dealloc];
}

- (void)showFromView:(UIView *)view routeSharingPolicy:(WebCore::RouteSharingPolicy)routeSharingPolicy routingContextUID:(NSString *)routingContextUID hasVideo:(BOOL)hasVideo
{
    static_assert(static_cast<size_t>(WebCore::RouteSharingPolicy::Default) == static_cast<size_t>(WKAirPlayRoutePickerRouteSharingPolicyDefault), "RouteSharingPolicy::Default is not WKAirPlayRoutePickerRouteSharingPolicyDefault as expected");
    static_assert(static_cast<size_t>(WebCore::RouteSharingPolicy::LongFormAudio) == static_cast<size_t>(WKAirPlayRoutePickerRouteSharingPolicyLongFormAudio), "RouteSharingPolicy::LongFormAudio is not WKAirPlayRoutePickerRouteSharingPolicyLongFormAudio as expected");
    static_assert(static_cast<size_t>(WebCore::RouteSharingPolicy::Independent) == static_cast<size_t>(WKAirPlayRoutePickerRouteSharingPolicyIndependent), "RouteSharingPolicy::Independent is not WKAirPlayRoutePickerRouteSharingPolicyIndependent as expected");
    static_assert(static_cast<size_t>(WebCore::RouteSharingPolicy::LongFormVideo) == static_cast<size_t>(WKAirPlayRoutePickerRouteSharingPolicyLongFormVideo), "RouteSharingPolicy::LongFormVideo is not WKAirPlayRoutePickerRouteSharingPolicyLongFormVideo as expected");
    if (_actionSheet)
        return;

    __block RetainPtr<MPAVRoutingController> routingController = adoptNS([allocMPAVRoutingControllerInstance() initWithName:@"WebKit - HTML media element showing AirPlay route picker"]);
    [routingController setDiscoveryMode:MPRouteDiscoveryModeDetailed];

    RetainPtr<MPMediaControlsConfiguration> configuration;
    if ([getMPMediaControlsConfigurationClass() instancesRespondToSelector:@selector(setSortByIsVideoRoute:)]) {
        configuration = adoptNS([allocMPMediaControlsConfigurationInstance() init]);
        configuration.get().sortByIsVideoRoute = hasVideo;
    }
    _actionSheet = adoptNS([allocMPMediaControlsViewControllerInstance() initWithConfiguration:configuration.get()]);

    if ([_actionSheet respondsToSelector:@selector(setOverrideRouteSharingPolicy:routingContextUID:)])
        [_actionSheet setOverrideRouteSharingPolicy:static_cast<WKAirPlayRoutePickerRouteSharingPolicy>(routeSharingPolicy) routingContextUID:routingContextUID];

    _actionSheet.get().didDismissHandler = ^ {
        [routingController setDiscoveryMode:MPRouteDiscoveryModeDisabled];
        routingController = nil;
        _actionSheet = nil;
    };

    auto viewControllerForPresentation = view._wk_viewControllerForFullScreenPresentation;
    [viewControllerForPresentation presentViewController:_actionSheet.get() animated:YES completion:nil];
}

@end

ALLOW_DEPRECATED_DECLARATIONS_END

#endif // PLATFORM(IOS_FAMILY) && ENABLE(AIRPLAY_PICKER)

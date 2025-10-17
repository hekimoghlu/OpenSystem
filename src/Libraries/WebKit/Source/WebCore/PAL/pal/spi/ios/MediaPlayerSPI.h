/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#import <wtf/Platform.h>

#if PLATFORM(IOS_FAMILY)

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#if USE(APPLE_INTERNAL_SDK)

#import <MediaPlayer/MPAVRoutingController.h>

#if !PLATFORM(WATCHOS) && !PLATFORM(APPLETV)
#import <MediaPlayer/MPMediaControlsConfiguration.h>
#import <MediaPlayer/MPMediaControlsViewController.h>
#endif

#else

NS_ASSUME_NONNULL_BEGIN

enum {
    MPRouteDiscoveryModeDisabled = 0,
    MPRouteDiscoveryModePresence = 1,
    MPRouteDiscoveryModeDetailed = 2,
};
typedef NSInteger MPRouteDiscoveryMode;

@interface MPAVRoutingController : NSObject
@end

@interface MPAVRoutingController ()
- (instancetype)initWithName:(NSString *)name;
@property (nonatomic, assign) MPRouteDiscoveryMode discoveryMode;
@end

#if !PLATFORM(WATCHOS) && !PLATFORM(APPLETV)
@interface MPMediaControlsViewController : UIViewController
@property (nonatomic, copy, nullable) void (^didDismissHandler)(void);
@end

@interface MPMediaControlsConfiguration : NSObject <NSSecureCoding, NSCopying>
@end
#endif

NS_ASSUME_NONNULL_END

#endif

#endif // PLATFORM(IOS_FAMILY)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#if USE(APPLE_INTERNAL_SDK)

#import <BackBoardServices/BKSAnimationFence.h>
#import <BackBoardServices/BKSAnimationFence_Private.h>
#import <BackBoardServices/BKSMousePointerService.h>

#else

#import "BaseBoardSPI.h"

@interface BKSAnimationFenceHandle : NSObject
- (mach_port_t)CAPort;
@end

@class BKSMousePointerDevice;

@protocol BKSMousePointerDeviceObserver <NSObject>
@optional
- (void)mousePointerDevicesDidConnect:(NSSet<BKSMousePointerDevice *> *)mousePointerDevices;
- (void)mousePointerDevicesDidDisconnect:(NSSet<BKSMousePointerDevice *> *)mousePointerDevices;
@end

@interface BKSMousePointerService : NSObject
+ (BKSMousePointerService *)sharedInstance;
- (id<BSInvalidatable>)addPointerDeviceObserver:(id<BKSMousePointerDeviceObserver>)observer;
@end

#endif // USE(APPLE_INTERNAL_SDK)

// Unfortunately, the following declarations need to be forward declared even when using the internal SDK,
// since the headers that define these symbols (BKSHIDEventKeyCommand.h and BKSHIDEventAttributes.h) include
// additional private headers that attempt to define macros, which conflict with other macros within WebKit
// (in particular, `kB` being defined in BrightnessSystemKeys.h, and Sizes.h in bmalloc).

typedef NS_OPTIONS(NSInteger, BKSKeyModifierFlags) {
    BKSKeyModifierShift = 1 << 17,
    BKSKeyModifierControl = 1 << 18,
    BKSKeyModifierAlternate = 1 << 19,
    BKSKeyModifierCommand = 1 << 20,
};

@interface BKSHIDEventBaseAttributes : NSObject
@end

@interface BKSHIDEventDigitizerAttributes : BKSHIDEventBaseAttributes
@property (nonatomic) BKSKeyModifierFlags activeModifiers;
@end

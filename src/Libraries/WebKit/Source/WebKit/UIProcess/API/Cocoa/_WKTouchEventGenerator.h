/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#if TARGET_OS_IPHONE

#import <CoreGraphics/CGGeometry.h>
#import <WebKit/WKFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@class UIWindow;
typedef struct __IOHIDEvent * IOHIDEventRef;

WK_CLASS_AVAILABLE(ios(13.0))
@interface _WKTouchEventGenerator : NSObject
+ (_WKTouchEventGenerator *)sharedTouchEventGenerator;

// The 'location' parameter is in screen coordinates, as used by IOHIDEvent.
- (void)touchDown:(CGPoint)location window:(UIWindow *)window completionBlock:(void (^)(void))completionBlock;
- (void)liftUp:(CGPoint)location window:(UIWindow *)window completionBlock:(void (^)(void))completionBlock;
- (void)moveToPoint:(CGPoint)location duration:(NSTimeInterval)seconds window:(UIWindow *)window completionBlock:(void (^)(void))completionBlock;

// Clients must call this method whenever a HID event is delivered to the UIApplication.
// It is used to detect when all synthesized touch events have been successfully delivered.
- (void)receivedHIDEvent:(IOHIDEventRef)event;
@end

NS_ASSUME_NONNULL_END

#endif // TARGET_OS_IPHONE

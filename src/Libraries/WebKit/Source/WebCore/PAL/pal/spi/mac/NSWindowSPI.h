/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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

#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSWindow_Private.h>

#else

#import <AppKit/NSWindow.h>

@interface NSWindow ()

- (id)_oldFirstResponderBeforeBecoming;
- (id)_newFirstResponderAfterResigning;
- (void)_setCursorForMouseLocation:(NSPoint)point;
- (void)exitFullScreenMode:(id)sender;
- (void)enterFullScreenMode:(id)sender;

enum {
    NSWindowChildOrderingPriorityPopover = 20,
};
- (NSInteger)_childWindowOrderingPriority;
@end

enum {
    NSSideUtilityWindowMask = 1 << 9,
    NSSmallWindowMask = 1 << 10,
    _NSCarbonWindowMask = 1 << 25,
};

#endif

extern NSNotificationName NSWindowDidOrderOffScreenNotification;
extern NSNotificationName NSWindowDidOrderOnScreenNotification;
extern NSNotificationName NSWindowWillOrderOffScreenNotification;
extern NSNotificationName NSWindowWillOrderOnScreenNotification;
extern NSNotificationName const _NSWindowDidChangeContentsHostedInLayerSurfaceNotification;

#endif // PLATFORM(MAC)

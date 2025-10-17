/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#if PLATFORM(MAC) && HAVE(TOUCH_BAR)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSCandidateListTouchBarItem_Private.h>
#import <AppKit/NSTextTouchBarItemController_WebKitSPI.h>
#import <AppKit/NSTouchBar_Private.h>

#endif

NS_ASSUME_NONNULL_BEGIN

#if !USE(APPLE_INTERNAL_SDK)

@interface NSTouchBar ()
@property (readonly, copy, nullable) NSArray<NSTouchBarItem *> *items;
@property (strong, nullable) NSTouchBarItem *escapeKeyReplacementItem;
@end

@interface NSTextTouchBarItemController : NSObject

@property (readonly, strong, nullable) NSColorPickerTouchBarItem *colorPickerItem;
@property (readonly, strong, nullable) NSSegmentedControl *textStyle;
@property (readonly, strong, nullable) NSSegmentedControl *textAlignments;
@property (nullable, strong) NSViewController *textListViewController;
@property BOOL usesNarrowTextStyleItem;

- (nullable NSTouchBarItem *)itemForIdentifier:(nullable NSString *)identifier;

@end

@interface NSCandidateListTouchBarItem ()
- (void)setCandidates:(NSArray *)candidates forSelectedRange:(NSRange)selectedRange inString:(nullable NSString *)string rect:(NSRect)rect view:(nullable NSView *)view completionHandler:(nullable void (^)(id acceptedCandidate))completionBlock;
@end

#endif // !USE(APPLE_INTERNAL_SDK)

APPKIT_EXTERN NSNotificationName const NSTouchBarWillEnterCustomization API_AVAILABLE(macos(10.12.2));
APPKIT_EXTERN NSNotificationName const NSTouchBarDidEnterCustomization API_AVAILABLE(macos(10.12.2));
APPKIT_EXTERN NSNotificationName const NSTouchBarWillExitCustomization API_AVAILABLE(macos(10.12.2));
APPKIT_EXTERN NSNotificationName const NSTouchBarDidExitCustomization API_AVAILABLE(macos(10.12.2));

NS_ASSUME_NONNULL_END

#endif // PLATFORM(MAC) && HAVE(TOUCH_BAR)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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

#import <AppKit/NSSharingService_Private.h>

@interface NSSharingServicePicker (Staging_r88855017)
- (NSMenuItem *)standardShareMenuItem;
@end

#else

#import <AppKit/NSSharingService.h>

typedef NS_ENUM(NSInteger, NSSharingServicePickerStyle) {
    NSSharingServicePickerStyleMenu = 0,
    NSSharingServicePickerStyleRollover = 1,
    NSSharingServicePickerStyleTextSelection = 2,
    NSSharingServicePickerStyleDataDetector = 3
} NS_ENUM_AVAILABLE_MAC(10_10);

@interface NSSharingServicePicker (Private)
@property NSSharingServicePickerStyle style;
- (NSMenuItem *)standardShareMenuItem;
- (NSMenu *)menu;
- (void)getMenuWithCompletion:(void(^)(NSMenu *))completion;
- (void)hide;
- (void)showPopoverRelativeToRect:(NSRect)rect ofView:(NSView *)view preferredEdge:(NSRectEdge)preferredEdge completion:(void (^)(NSSharingService *))completion;
@end

#endif

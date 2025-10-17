/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#import <AppKit/NSColor.h>

#if PLATFORM(MAC) && USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSColor_Private.h>
#import <AppKit/NSColor_UserAccent.h>

#else

@interface NSColor ()
+ (NSColor *)systemRedColor;
+ (NSColor *)systemGreenColor;
+ (NSColor *)systemBlueColor;
+ (NSColor *)systemOrangeColor;
+ (NSColor *)systemYellowColor;
+ (NSColor *)systemBrownColor;
+ (NSColor *)systemPinkColor;
+ (NSColor *)systemPurpleColor;
+ (NSColor *)systemGrayColor;
+ (NSColor *)linkColor;
+ (NSColor *)findHighlightColor;
+ (NSColor *)placeholderTextColor;
+ (NSColor *)containerBorderColor;
@end

typedef NS_ENUM(NSInteger, NSUserAccentColor) {
    NSUserAccentColorRed = 0,
    NSUserAccentColorOrange,
    NSUserAccentColorYellow,
    NSUserAccentColorGreen,
    NSUserAccentColorBlue,
    NSUserAccentColorPurple,
    NSUserAccentColorPink,

    NSUserAccentColorNoColor = -1,
};

extern "C" NSUserAccentColor NSColorGetUserAccentColor(void);

#endif

// FIXME: Remove staging when AppKit without tertiary-fill is not used anymore; see rdar://108340604.
#if HAVE(NSCOLOR_FILL_COLOR_HIERARCHY)
@interface NSColor (Staging_104764768)
@property (class, strong, readonly) NSColor *tertiarySystemFillColor;
@end
#endif

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#if PLATFORM(MAC)

#import <AppKit/AppKit.h>

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSGraphicsContextPrivate.h>
#import <AppKit/NSGraphics_Private.h>

#else

#import <pal/spi/cg/CoreGraphicsSPI.h>

@interface NSCGSContext : NSGraphicsContext {
    CGContextRef _cgsContext;
}
@end

@interface NSWindowGraphicsContext : NSCGSContext {
    CGSWindowID _cgsWindowID;
}
@end

typedef NS_OPTIONS(NSUInteger, NSMenuBackgroundFlags) {
    NSMenuBackgroundPopupMenu = 0x200
};

APPKIT_EXTERN void NSDrawMenuBackground(NSRect bounds, NSRect clipRect, NSMenuBackgroundFlags);

#endif

WTF_EXTERN_C_BEGIN

BOOL NSInitializeCGFocusRingStyleForTime(NSFocusRingPlacement, CGFocusRingStyle*, NSTimeInterval);

WTF_EXTERN_C_END

#endif // PLATFORM(MAC)

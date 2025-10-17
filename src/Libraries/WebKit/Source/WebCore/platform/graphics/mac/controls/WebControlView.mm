/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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
#import "WebControlView.h"

#if PLATFORM(MAC)

#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>
#import <pal/spi/mac/NSCellSPI.h>
#import <pal/spi/mac/NSImageSPI.h>
#import <pal/spi/mac/NSServicesRolloverButtonCellSPI.h>
#import <pal/spi/mac/NSSharingServicePickerSPI.h>
#import <pal/spi/mac/NSTextFieldCellSPI.h>

@implementation WebControlWindow

static BOOL _hasKeyAppearance;

+ (BOOL)hasKeyAppearance
{
    return _hasKeyAppearance;
}

+ (void)setHasKeyAppearance:(BOOL)newHasKeyAppearance
{
    _hasKeyAppearance = newHasKeyAppearance;
}

- (BOOL)hasKeyAppearance
{
    return _hasKeyAppearance;
}

- (BOOL)isKeyWindow
{
    return _hasKeyAppearance;
}

// FIXME: rdar://105250168 - Remove this code once viewless NSCell drawing is complete.
- (BOOL)_needsToResetDragMargins
{
    return false;
}

- (void)_setNeedsToResetDragMargins:(BOOL)needs
{
}
@end

@implementation WebControlView {
    RetainPtr<WebControlWindow> _window;
}

static NSRect _clipBounds;

+ (NSRect)clipBounds
{
    return _clipBounds;
}

+ (void)setClipBounds:(NSRect)newClipBounds
{
    _clipBounds = newClipBounds;
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    // Using defer:YES prevents us from wasting any window server resources for this window, since we're not actually
    // going to draw into it. The other arguments match what you get when calling -[NSWindow init].
    _window = adoptNS([[WebControlWindow alloc] initWithContentRect:NSMakeRect(100, 100, 100, 100) styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:YES]);

    return self;
}

- (NSWindow *)window
{
    return _window.get();
}

- (BOOL)isFlipped
{
    return YES;
}

- (NSText *)currentEditor
{
    return nil;
}

- (BOOL)_automaticFocusRingDisabled
{
    return YES;
}

- (NSRect)_focusRingVisibleRect
{
    if (NSIsEmptyRect(_clipBounds))
        return [self visibleRect];
    return _clipBounds;
}

- (NSView *)_focusRingClipAncestor
{
    return self;
}

- (NSWindow *)_viewRoot
{
    return _window.get();
}

- (void)addSubview:(NSView *)subview
{
    // By doing nothing in this method we forbid controls from adding subviews.
    // This tells AppKit to not use layer-backed animation for control rendering.
    UNUSED_PARAM(subview);
}

@end

@implementation WebControlTextFieldCell

- (CFDictionaryRef)_adjustedCoreUIDrawOptionsForDrawingBordersOnly:(CFDictionaryRef)defaultOptions
{
    // Dark mode controls don't have borders, just a semi-transparent background of shadows.
    // In the dark mode case we can't disable borders, or we will not paint anything for the control.
    NSAppearanceName appearance = [[NSAppearance currentDrawingAppearance] bestMatchFromAppearancesWithNames:@[ NSAppearanceNameAqua, NSAppearanceNameDarkAqua ]];
    if ([appearance isEqualToString:NSAppearanceNameDarkAqua])
        return defaultOptions;

    // FIXME: This is a workaround for <rdar://problem/11385461>. When that bug is resolved, we should remove this code,
    // as well as the internal method overrides below.
    auto coreUIDrawOptions = adoptCF(CFDictionaryCreateMutableCopy(NULL, 0, defaultOptions));
    CFDictionarySetValue(coreUIDrawOptions.get(), CFSTR("borders only"), kCFBooleanTrue);
    return coreUIDrawOptions.autorelease();
}

- (CFDictionaryRef)_coreUIDrawOptionsWithFrame:(NSRect)cellFrame inView:(NSView *)controlView includeFocus:(BOOL)includeFocus
{
    return [self _adjustedCoreUIDrawOptionsForDrawingBordersOnly:[super _coreUIDrawOptionsWithFrame:cellFrame inView:controlView includeFocus:includeFocus]];
}

- (CFDictionaryRef)_coreUIDrawOptionsWithFrame:(NSRect)cellFrame inView:(NSView *)controlView includeFocus:(BOOL)includeFocus maskOnly:(BOOL)maskOnly
{
    return [self _adjustedCoreUIDrawOptionsForDrawingBordersOnly:[super _coreUIDrawOptionsWithFrame:cellFrame inView:controlView includeFocus:includeFocus maskOnly:maskOnly]];
}

@end

#endif // PLATFORM(MAC)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#import "WebCoreFullScreenWarningView.h"

#if PLATFORM(MAC) && ENABLE(FULLSCREEN_API)

#import "LocalizedStrings.h"
#import <wtf/text/WTFString.h>

static const CGFloat WarningViewTextWhite = 0.9;
static const CGFloat WarningViewTextAlpha = 1;
static const CGFloat WarningViewTextSize = 48;
static const CGFloat WarningViewPadding = 20;
static const CGFloat WarningViewCornerRadius = 10;
static const CGFloat WarningViewBorderWhite = 0.9;
static const CGFloat WarningViewBorderAlpha = 0.2;
static const CGFloat WarningViewBackgroundWhite = 0.1;
static const CGFloat WarningViewBackgroundAlpha = 0.9;
static const CGFloat WarningViewShadowWhite = 0.1;
static const CGFloat WarningViewShadowAlpha = 1;
static const NSSize WarningViewShadowOffset = {0, -2};
static const CGFloat WarningViewShadowRadius = 5;

@implementation WebCoreFullScreenWarningView

- (id)initWithTitle:(NSString*)title
{
    self = [super initWithFrame:NSZeroRect];
    if (!self)
        return nil;

    [self setAutoresizingMask:(NSViewMinXMargin | NSViewMaxXMargin | NSViewMinYMargin | NSViewMaxYMargin)];
    [self setBoxType:NSBoxCustom];
    [self setTitlePosition:NSNoTitle];

    _textField = adoptNS([[NSTextField alloc] initWithFrame:NSZeroRect]);
    [_textField setEditable:NO];
    [_textField setSelectable:NO];
    [_textField setBordered:NO];
    [_textField setDrawsBackground:NO];

    NSFont* textFont = [NSFont boldSystemFontOfSize:WarningViewTextSize];
    NSColor* textColor = [NSColor colorWithCalibratedWhite:WarningViewTextWhite alpha:WarningViewTextAlpha];
    RetainPtr<NSDictionary> attributes = adoptNS([[NSDictionary alloc] initWithObjectsAndKeys:
                                                  textFont, NSFontAttributeName,
                                                  textColor, NSForegroundColorAttributeName,
                                                  nil]);
    RetainPtr<NSAttributedString> text = adoptNS([[NSAttributedString alloc] initWithString:title attributes:attributes.get()]);
    [_textField setAttributedStringValue:text.get()];
    [_textField sizeToFit];
    NSRect textFieldFrame = [_textField frame];
    NSSize frameSize = textFieldFrame.size;
    frameSize.width += WarningViewPadding * 2;
    frameSize.height += WarningViewPadding * 2;
    [self setFrameSize:frameSize];

    textFieldFrame.origin = NSMakePoint(
        (frameSize.width - textFieldFrame.size.width) / 2,
        (frameSize.height - textFieldFrame.size.height) / 2);

    // Offset the origin by the font's descender, to center the text field about the baseline:
    textFieldFrame.origin.y += [[_textField font] descender];

    [_textField setFrame:NSIntegralRect(textFieldFrame)];
    [self addSubview:_textField.get()];

    NSColor* backgroundColor = [NSColor colorWithCalibratedWhite:WarningViewBackgroundWhite alpha:WarningViewBackgroundAlpha];
    [self setFillColor:backgroundColor];
    [self setCornerRadius:WarningViewCornerRadius];

    NSColor* borderColor = [NSColor colorWithCalibratedWhite:WarningViewBorderWhite alpha:WarningViewBorderAlpha];
    [self setBorderColor:borderColor];

    RetainPtr<NSShadow> shadow = adoptNS([[NSShadow alloc] init]);
    RetainPtr<NSColor> shadowColor = [NSColor colorWithCalibratedWhite:WarningViewShadowWhite alpha:WarningViewShadowAlpha];
    [shadow setShadowColor:shadowColor.get()];
    [shadow setShadowOffset:WarningViewShadowOffset];
    [shadow setShadowBlurRadius:WarningViewShadowRadius];
    [self setShadow:shadow.get()];

    return self;
}
@end

#endif // PLATFORM(MAC) && ENABLE(FULLSCREEN_API)

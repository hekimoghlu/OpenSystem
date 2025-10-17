/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#import "WebCoreFullScreenPlaceholderView.h"

#if PLATFORM(MAC)

#import "LocalizedStrings.h"
#import "WebCoreFullScreenWarningView.h"
#import <wtf/text/WTFString.h>

@implementation WebCoreFullScreenPlaceholderView

- (id)initWithFrame:(NSRect)frameRect
{
    self = [super initWithFrame:frameRect];
    if (!self)
        return nil;

    self.wantsLayer = YES;
    self.autoresizesSubviews = YES;
    self.layerContentsPlacement = NSViewLayerContentsPlacementScaleProportionallyToFit;
    self.layerContentsRedrawPolicy = NSViewLayerContentsRedrawNever;

    _effectView = adoptNS([[NSVisualEffectView alloc] initWithFrame:frameRect]);
    _effectView.get().wantsLayer = YES;
    _effectView.get().autoresizesSubviews = YES;
    _effectView.get().autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _effectView.get().blendingMode = NSVisualEffectBlendingModeWithinWindow;
    _effectView.get().hidden = YES;
    _effectView.get().state = NSVisualEffectStateActive;
    _effectView.get().material = NSVisualEffectMaterialPopover;
    [self addSubview:_effectView.get()];

    _exitWarning = adoptNS([[NSTextField alloc] initWithFrame:NSZeroRect]);
    _exitWarning.get().autoresizingMask = NSViewMinXMargin | NSViewMaxXMargin | NSViewMinYMargin | NSViewMaxYMargin;
    _exitWarning.get().bordered = NO;
    _exitWarning.get().drawsBackground = NO;
    _exitWarning.get().editable = NO;
    _exitWarning.get().font = [NSFont systemFontOfSize:27];
    _exitWarning.get().selectable = NO;
    _exitWarning.get().stringValue = WebCore::clickToExitFullScreenText();
    _exitWarning.get().textColor = [NSColor tertiaryLabelColor];
    [_exitWarning sizeToFit];

    NSRect warningFrame = [_exitWarning frame];
    warningFrame.origin = NSMakePoint((frameRect.size.width - warningFrame.size.width) / 2, frameRect.size.height / 2);
    _exitWarning.get().frame = warningFrame;
    [_effectView addSubview:_exitWarning.get()];

    return self;
}

- (NSResponder *)target
{
    return _target.get().get();
}

- (void)setTarget:(NSResponder *)target
{
    _target = target;
}

@dynamic contents;

- (void)setContents:(id)contents
{
    [[self layer] setContents:contents];
}

- (id)contents
{
    return [[self layer] contents];
}

- (void)setExitWarningVisible:(BOOL)visible
{
    [_effectView setHidden:!visible];
}

- (void)mouseDown:(NSEvent *)event
{
    UNUSED_PARAM(event);
    [_target cancelOperation:self];
}

@end

#endif // !PLATFORM(IOS_FAMILY)

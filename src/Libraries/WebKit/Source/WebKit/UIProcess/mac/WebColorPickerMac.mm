/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#import "WebColorPickerMac.h"

#if USE(APPKIT)

#import <WebCore/Color.h>
#import <WebCore/ColorMac.h>
#import <pal/spi/mac/NSColorWellSPI.h>
#import <pal/spi/mac/NSPopoverColorWellSPI.h>
#import <pal/spi/mac/NSPopoverSPI.h>
#import <wtf/WeakObjCPtr.h>

static const size_t maxColorSuggestions = 12;
static const CGFloat colorPickerMatrixNumColumns = 12.0;
static const CGFloat colorPickerMatrixBorderWidth = 1.0;

// FIXME: <rdar://problem/41173525> We should not have to track changes in NSPopoverColorWell's implementation.
static const CGFloat colorPickerMatrixSwatchWidth = 13.0;

@protocol WKPopoverColorWellDelegate <NSObject>
- (void)didClosePopover;
@end

@interface WKPopoverColorWell : NSPopoverColorWell {
    RetainPtr<NSColorList> _suggestedColors;
    WeakObjCPtr<id <WKPopoverColorWellDelegate>> _webDelegate;
}

@property (nonatomic, weak) id<WKPopoverColorWellDelegate> webDelegate;

- (void)setSuggestedColors:(NSColorList *)suggestedColors;
@end

@interface WKColorPopoverMac : NSObject<WKColorPickerUIMac, WKPopoverColorWellDelegate, NSWindowDelegate> {
@private
    BOOL _lastChangedByUser;
    WebKit::WebColorPickerMac *_picker;
    RetainPtr<WKPopoverColorWell> _popoverWell;
}
- (id)initWithFrame:(const WebCore::IntRect &)rect inView:(NSView *)view;
@end

namespace WebKit {

Ref<WebColorPickerMac> WebColorPickerMac::create(WebColorPicker::Client* client, const WebCore::Color& initialColor, const WebCore::IntRect& rect, WebKit::ColorControlSupportsAlpha supportsAlpha, Vector<WebCore::Color>&& suggestions, NSView *view)
{
    return adoptRef(*new WebColorPickerMac(client, initialColor, rect, supportsAlpha, WTFMove(suggestions), view));
}

WebColorPickerMac::~WebColorPickerMac()
{
    if (m_colorPickerUI) {
        [m_colorPickerUI invalidate];
        m_colorPickerUI = nil;
    }
}

WebColorPickerMac::WebColorPickerMac(WebColorPicker::Client* client, const WebCore::Color& initialColor, const WebCore::IntRect& rect, WebKit::ColorControlSupportsAlpha supportsAlpha, Vector<WebCore::Color>&& suggestions, NSView *view)
    : WebColorPicker(client)
    , m_supportsAlpha(supportsAlpha)
    , m_suggestions(WTFMove(suggestions))
{
    m_colorPickerUI = adoptNS([[WKColorPopoverMac alloc] initWithFrame:rect inView:view]);
}

void WebColorPickerMac::endPicker()
{
    [m_colorPickerUI invalidate];
    m_colorPickerUI = nil;
    WebColorPicker::endPicker();
}

void WebColorPickerMac::setSelectedColor(const WebCore::Color& color)
{
    if (!client() || !m_colorPickerUI)
        return;
    
    [m_colorPickerUI setColor:cocoaColor(color).get()];
}

void WebColorPickerMac::didChooseColor(const WebCore::Color& color)
{
    if (CheckedPtr client = this->client())
        client->didChooseColor(color);
}

void WebColorPickerMac::showColorPicker(const WebCore::Color& color)
{
    if (!client())
        return;

    [m_colorPickerUI setAndShowPicker:this withColor:cocoaColor(color).get() supportsAlpha:m_supportsAlpha suggestions:WTFMove(m_suggestions)];
}

} // namespace WebKit

@implementation WKPopoverColorWell

+ (NSPopover *)_colorPopoverCreateIfNecessary:(BOOL)forceCreation
{
    static NeverDestroyed<RetainPtr<NSPopover>> colorPopover;
    if (forceCreation) {
        auto popover = adoptNS([[NSPopover alloc] init]);
        [popover _setRequiresCorrectContentAppearance:YES];
        [popover setBehavior:NSPopoverBehaviorTransient];

        auto controller = adoptNS([[NSClassFromString(@"NSColorPopoverController") alloc] init]);
        [popover setContentViewController:controller.get()];
        [controller setPopover:popover.get()];

        colorPopover.get() = WTFMove(popover);
    }

    return colorPopover.get().get();
}

- (id <WKPopoverColorWellDelegate>)webDelegate
{
    return _webDelegate.getAutoreleased();
}

- (void)setWebDelegate:(id <WKPopoverColorWellDelegate>)webDelegate
{
    _webDelegate = webDelegate;
}

- (void)_showPopover
{
    NSPopover *popover = [[self class] _colorPopoverCreateIfNecessary:YES];
    popover.delegate = self;

    [self deactivate];

    // Deactivate previous NSPopoverColorWell
    NSColorWell *owner = [NSColorWell _exclusiveColorPanelOwner];
    if ([owner isKindOfClass:[NSPopoverColorWell class]])
        [owner deactivate];

    NSColorPopoverController *controller = (NSColorPopoverController *)[popover contentViewController];
    controller.delegate = self;

    if (_suggestedColors) {
        NSUInteger numColors = [[_suggestedColors allKeys] count];
        CGFloat swatchWidth = (colorPickerMatrixNumColumns * colorPickerMatrixSwatchWidth + (colorPickerMatrixNumColumns * colorPickerMatrixBorderWidth - numColors)) / numColors;
        CGFloat swatchHeight = colorPickerMatrixSwatchWidth;

        // topBarMatrixView cannot be accessed until view has been loaded
        if (!controller.isViewLoaded)
            [controller loadView];

        NSColorPickerMatrixView *topMatrix = controller.topBarMatrixView;
        [topMatrix setNumberOfColumns:numColors];
        [topMatrix setSwatchSize:NSMakeSize(swatchWidth, swatchHeight)];
        [topMatrix setColorList:_suggestedColors.get()];
    }

    [self activate:YES];
    [popover showRelativeToRect:self.bounds ofView:self preferredEdge:NSMinYEdge];
}

- (void)popoverDidClose:(NSNotification *)notification {
    [self.webDelegate didClosePopover];
}

- (NSView *)hitTest:(NSPoint)point
{
    return nil;
}

- (void)setSuggestedColors:(NSColorList *)suggestedColors
{
    _suggestedColors = suggestedColors;
}

@end

@implementation WKColorPopoverMac
- (id)initWithFrame:(const WebCore::IntRect &)rect inView:(NSView *)view
{
    if(!(self = [super init]))
        return self;

    _popoverWell = adoptNS([[WKPopoverColorWell alloc] initWithFrame:[view convertRect:NSRectFromCGRect(rect) toView:nil]]);
    if (!_popoverWell)
        return self;

    [_popoverWell setAlphaValue:0.0];
    [[view window].contentView addSubview:_popoverWell.get()];

    return self;
}

- (void)setAndShowPicker:(WebKit::WebColorPickerMac*)picker withColor:(NSColor *)color supportsAlpha:(WebKit::ColorControlSupportsAlpha)supportsAlpha suggestions:(Vector<WebCore::Color>&&)suggestions
{
    _picker = picker;

    [_popoverWell setTarget:self];
    [_popoverWell setWebDelegate:self];
    [_popoverWell setAction:@selector(didChooseColor:)];
    [_popoverWell setColor:color];
#if HAVE(NSCOLORWELL_SUPPORTS_ALPHA)
    [_popoverWell setSupportsAlpha:supportsAlpha == WebKit::ColorControlSupportsAlpha::Yes];
#endif

    RetainPtr<NSColorList> suggestedColors;
    if (suggestions.size()) {
        suggestedColors = adoptNS([[NSColorList alloc] init]);
        for (size_t i = 0; i < std::min(suggestions.size(), maxColorSuggestions); i++)
            [suggestedColors insertColor:cocoaColor(suggestions.at(i)).get() key:@(i).stringValue atIndex:i];
    }

    [_popoverWell setSuggestedColors:suggestedColors.get()];
    [_popoverWell _showPopover];

    [[NSColorPanel sharedColorPanel] setDelegate:self];
    
    _lastChangedByUser = YES;
}

- (void)invalidate
{
    [_popoverWell removeFromSuperviewWithoutNeedingDisplay];
    [_popoverWell setTarget:nil];
    [_popoverWell setAction:nil];
    [_popoverWell deactivate];
    
    _popoverWell = nil;
    _picker = nil;

    NSColorPanel *panel = [NSColorPanel sharedColorPanel];
    if (panel.delegate == self) {
        panel.delegate = nil;
        [panel close];
    }
}

- (void)windowWillClose:(NSNotification *)notification
{
    if (!_picker)
        return;

    if (notification.object == [NSColorPanel sharedColorPanel]) {
        _lastChangedByUser = YES;
        _picker->endPicker();
    }
}

- (void)didChooseColor:(id)sender
{
    if (sender != _popoverWell)
        return;

    // Handle the case where the <input type='color'> value is programmatically set.
    if (!_lastChangedByUser) {
        _lastChangedByUser = YES;
        return;
    }

    _picker->didChooseColor(WebCore::colorFromCocoaColor([_popoverWell color]));
}

- (void)setColor:(NSColor *)color
{
    _lastChangedByUser = NO;
    [_popoverWell setColor:color];
}

- (void)didClosePopover
{
    if (!_picker)
        return;

    if (![NSColorPanel sharedColorPanel].isVisible)
        _picker->endPicker();
}

@end

#endif // USE(APPKIT)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#import "WebNodeHighlight.h"
#import "WebNodeHighlightView.h"
#import "WebNSViewExtras.h"

#import <WebCore/InspectorController.h>
#import <wtf/Assertions.h>

#if PLATFORM(IOS_FAMILY)
#import "WebFramePrivate.h"
#import "WebHTMLView.h"
#import "WebView.h"
#import <WebCore/WAKWindow.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#endif

using namespace WebCore;

#if !PLATFORM(IOS_FAMILY)
@interface WebNodeHighlight (FileInternal)
- (NSRect)_computeHighlightWindowFrame;
- (void)_repositionHighlightWindow;
@end
#endif

#if PLATFORM(IOS_FAMILY)
@implementation WebHighlightLayer

- (id)initWithHighlightView:(WebNodeHighlightView *)view webView:(WebView *)webView
{
    self = [super init];
    if (!self)
        return nil;
    _view = view;
    _webView = webView;
    return self;
}

- (void)layoutSublayers
{
    CGFloat documentScale = [[[_webView mainFrame] documentView] scale];
    [self setTransform:CATransform3DMakeScale(documentScale, documentScale, 1.0)];

    [_view layoutSublayers:self];
}

- (id<CAAction>)actionForKey:(NSString *)key
{
    return nil; // Disable all default actions.
}

@end
#endif

@implementation WebNodeHighlight

- (id)initWithTargetView:(NSView *)targetView inspectorController:(NakedPtr<InspectorController>)inspectorController
{
    self = [super init];
    if (!self)
        return nil;

    _targetView = [targetView retain];
    _inspectorController = inspectorController;

#if !PLATFORM(IOS_FAMILY)
    int styleMask = NSWindowStyleMaskBorderless;
    NSRect contentRect = [NSWindow contentRectForFrameRect:[self _computeHighlightWindowFrame] styleMask:styleMask];
    _highlightWindow = [[NSWindow alloc] initWithContentRect:contentRect styleMask:styleMask backing:NSBackingStoreBuffered defer:NO];
    [_highlightWindow setBackgroundColor:[NSColor clearColor]];
    [_highlightWindow setOpaque:NO];
    [_highlightWindow setIgnoresMouseEvents:YES];
    [_highlightWindow setReleasedWhenClosed:NO];

    _highlightView = [[WebNodeHighlightView alloc] initWithWebNodeHighlight:self];
    [_highlightWindow setContentView:_highlightView];
    [_highlightView release];
#else
    ASSERT([_targetView isKindOfClass:[WebView class]]);
    WebView *webView = (WebView *)targetView;

    _highlightView = [[WebNodeHighlightView alloc] initWithWebNodeHighlight:self];
    _highlightLayer = [[WebHighlightLayer alloc] initWithHighlightView:_highlightView webView:webView];
    [_highlightLayer setContentsScale:[[_targetView window] screenScale]]; // HiDPI.
    [_highlightLayer setCanDrawConcurrently:NO];
#endif

    return self;
}

- (void)dealloc
{
#if !PLATFORM(IOS_FAMILY)
    ASSERT(!_highlightWindow);
#else
    ASSERT(!_highlightLayer);
#endif
    ASSERT(!_targetView);
    ASSERT(!_highlightView);

    [super dealloc];
}

- (void)attach
{
    ASSERT(_targetView);
    ASSERT([_targetView window]);

#if !PLATFORM(IOS_FAMILY)
    ASSERT(_highlightWindow);

    if (!_highlightWindow || !_targetView || ![_targetView window])
        return;

    [[_targetView window] addChildWindow:_highlightWindow ordered:NSWindowAbove];

    // Observe both frame-changed and bounds-changed notifications because either one could leave
    // the highlight incorrectly positioned with respect to the target view. We need to do this for
    // the entire superview hierarchy to handle scrolling, bars coming and going, etc. 
    // (without making concrete assumptions about the view hierarchy).
    NSNotificationCenter *notificationCenter = [NSNotificationCenter defaultCenter];
    for (NSView *v = _targetView; v; v = [v superview]) {
        [notificationCenter addObserver:self selector:@selector(_repositionHighlightWindow) name:NSViewFrameDidChangeNotification object:v];
        [notificationCenter addObserver:self selector:@selector(_repositionHighlightWindow) name:NSViewBoundsDidChangeNotification object:v];
    }
#else
    ASSERT(_highlightLayer);

    WAKWindow *window = [_targetView window];
    [[window hostLayer] addSublayer:_highlightLayer];
    [self setNeedsDisplay];
#endif

    if (_delegate && [_delegate respondsToSelector:@selector(didAttachWebNodeHighlight:)])
        [_delegate didAttachWebNodeHighlight:self];
}

- (id)delegate
{
    return _delegate;
}

- (void)detach
{
#if !PLATFORM(IOS_FAMILY)
    if (!_highlightWindow) {
#else
    if (!_highlightLayer) {
#endif
        ASSERT(!_targetView);
        return;
    }

    if (_delegate && [_delegate respondsToSelector:@selector(willDetachWebNodeHighlight:)])
        [_delegate willDetachWebNodeHighlight:self];

#if !PLATFORM(IOS_FAMILY)
    NSNotificationCenter *notificationCenter = [NSNotificationCenter defaultCenter];
    [notificationCenter removeObserver:self name:NSViewFrameDidChangeNotification object:nil];
    [notificationCenter removeObserver:self name:NSViewBoundsDidChangeNotification object:nil];

    [[_highlightWindow parentWindow] removeChildWindow:_highlightWindow];
    [_highlightWindow close];

    [_highlightWindow release];
    _highlightWindow = nil;
#else
    [_highlightLayer removeFromSuperlayer];
    [_highlightLayer release];
    _highlightLayer = nil;
#endif

    [_targetView release];
    _targetView = nil;

    // We didn't retain _highlightView, but we do need to tell it to forget about us, so it doesn't
    // try to send our delegate messages after we've been dealloc'ed, e.g.
    [_highlightView detachFromWebNodeHighlight];
#if PLATFORM(IOS_FAMILY)
    // iOS did retain the highlightView, and we should release it here.
    [_highlightView release];
#endif
    _highlightView = nil;
}

- (WebNodeHighlightView *)highlightView
{
    return _highlightView;
}

- (void)setDelegate:(id)delegate
{
    // The delegate is not retained, as usual in Cocoa.
    _delegate = delegate;
}

#if !PLATFORM(IOS_FAMILY)
- (void)setNeedsUpdateInTargetViewRect:(NSRect)rect
{
    ASSERT(_targetView);

    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [[_targetView window] disableScreenUpdatesUntilFlush];
    ALLOW_DEPRECATED_DECLARATIONS_END

    // Mark the whole highlight view as needing display since we don't know what areas
    // need updated, since the highlight can be larger than the element to show margins.
    [_highlightView setNeedsDisplay:YES];

    // Redraw highlight view immediately so it updates in sync with the target view.
    // This is especially visible when resizing the window, scrolling or with DHTML.
    [_highlightView displayIfNeeded];
}
#else
- (void)setNeedsDisplay
{
    [_highlightLayer setNeedsLayout];
    [_highlightLayer setNeedsDisplay];
    [_highlightLayer displayIfNeeded];
}
#endif

- (NSView *)targetView
{
    return _targetView;
}

- (NakedPtr<InspectorController>)inspectorController
{
    return _inspectorController;
}

@end

#if !PLATFORM(IOS_FAMILY)
@implementation WebNodeHighlight (FileInternal)

- (NSRect)_computeHighlightWindowFrame
{
    ASSERT(_targetView);
    ASSERT([_targetView window]);

    NSRect highlightWindowFrame = [_targetView convertRect:[_targetView visibleRect] toView:nil];
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    highlightWindowFrame.origin = [[_targetView window] convertBaseToScreen:highlightWindowFrame.origin];
ALLOW_DEPRECATED_DECLARATIONS_END

    return highlightWindowFrame;
}

- (void)_repositionHighlightWindow
{
    // This assertion fires in cases where a new tab is created while the highlight
    // is showing (<http://bugs.webkit.org/show_bug.cgi?id=14254>)
    ASSERT([_targetView window]);
    
    // Until that bug is fixed, bail out to avoid worse problems where the highlight
    // moves to a nonsense location.
    if (![_targetView window])
        return;

    // Disable screen updates so the highlight moves in sync with the view.
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [[_targetView window] disableScreenUpdatesUntilFlush];
    ALLOW_DEPRECATED_DECLARATIONS_END

    [_highlightWindow setFrame:[self _computeHighlightWindowFrame] display:YES];
}

@end
#endif

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#pragma once

#import "config.h"
#import "WKRevealItemPresenter.h"

#if PLATFORM(MAC) && ENABLE(REVEAL)

#import "WebViewImpl.h"
#import <wtf/RetainPtr.h>
#import <wtf/WeakPtr.h>
#import <pal/cocoa/RevealSoftLink.h>

@interface WKRevealItemPresenter () <RVPresenterHighlightDelegate>
@end

@implementation WKRevealItemPresenter {
    WeakPtr<WebKit::WebViewImpl> _impl;
    RetainPtr<RVPresenter> _presenter;
    RetainPtr<RVPresentingContext> _presentingContext;
    RetainPtr<RVItem> _item;
    CGRect _frameInView;
    CGPoint _menuLocationInView;
    BOOL _isHighlightingItem;
}

- (instancetype)initWithWebViewImpl:(const WebKit::WebViewImpl&)impl item:(RVItem *)item frame:(CGRect)frameInView menuLocation:(CGPoint)menuLocationInView
{
    if (!(self = [super init]))
        return nil;

    _impl = impl;
    _presenter = adoptNS([PAL::allocRVPresenterInstance() init]);
    _presentingContext = adoptNS([PAL::allocRVPresentingContextInstance() initWithPointerLocationInView:menuLocationInView inView:impl.view() highlightDelegate:self]);
    _item = item;
    _frameInView = frameInView;
    _menuLocationInView = menuLocationInView;
    return self;
}

- (void)showContextMenu
{
    if (!_impl)
        return;

    auto view = _impl->view();
    if (!view)
        return;

    auto menuItems = retainPtr([_presenter menuItemsForItem:_item.get() documentContext:nil presentingContext:_presentingContext.get() options:nil]);
    if (![menuItems count])
        return;

    auto menu = adoptNS([[NSMenu alloc] initWithTitle:emptyString()]);
    [menu setAutoenablesItems:NO];
    [menu setItemArray:menuItems.get()];

    auto clickLocationInWindow = [view convertPoint:_menuLocationInView toView:nil];
    NSEvent *event = [NSEvent mouseEventWithType:NSEventTypeLeftMouseDown location:clickLocationInWindow modifierFlags:0 timestamp:0 windowNumber:view.window.windowNumber context:0 eventNumber:0 clickCount:1 pressure:1];
    [NSMenu popUpContextMenu:menu.get() withEvent:event forView:view];

    [self _callDidFinishPresentationIfNeeded];
}

- (void)_callDidFinishPresentationIfNeeded
{
    if (!_impl || _isHighlightingItem)
        return;

    _impl->didFinishPresentation(self);
}

#pragma mark - RVPresenterHighlightDelegate

- (NSArray<NSValue *> *)revealContext:(RVPresentingContext *)context rectsForItem:(RVItem *)item
{
    return @[ [NSValue valueWithRect:_frameInView] ];
}

- (BOOL)revealContext:(RVPresentingContext *)context shouldUseDefaultHighlightForItem:(RVItem *)item
{
    return self.shouldUseDefaultHighlight;
}

- (void)revealContext:(RVPresentingContext *)context startHighlightingItem:(RVItem *)item
{
    _isHighlightingItem = YES;
}

- (void)revealContext:(RVPresentingContext *)context stopHighlightingItem:(RVItem *)item
{
    _isHighlightingItem = NO;

    [self _callDidFinishPresentationIfNeeded];
}

@end

#endif // PLATFORM(MAC) && ENABLE(REVEAL)

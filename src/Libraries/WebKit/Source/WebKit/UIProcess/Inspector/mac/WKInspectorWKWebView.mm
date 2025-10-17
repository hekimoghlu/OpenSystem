/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#import "WKInspectorWKWebView.h"

#if PLATFORM(MAC)

#import "WKInspectorPrivateMac.h"
#import <wtf/WeakObjCPtr.h>

@implementation WKInspectorWKWebView {
    WeakObjCPtr<id <WKInspectorWKWebViewDelegate>> _inspectorWKWebViewDelegate;
}

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self name:NSWindowDidBecomeKeyNotification object:nil];

    [super dealloc];
}

- (NSRect)_opaqueRectForWindowMoveWhenInTitlebar
{
    // This convinces AppKit to allow window moves when clicking anywhere in the titlebar (top 22pt)
    // when this view's contents cover the entire window's contents, including the titlebar.
    return NSZeroRect;
}

- (NSInteger)tag
{
    return WKInspectorViewTag;
}

- (id <WKInspectorWKWebViewDelegate>)inspectorWKWebViewDelegate
{
    return _inspectorWKWebViewDelegate.getAutoreleased();
}

- (void)setInspectorWKWebViewDelegate:(id <WKInspectorWKWebViewDelegate>)delegate
{
    if (!!_inspectorWKWebViewDelegate)
        [[NSNotificationCenter defaultCenter] removeObserver:self name:NSWindowDidBecomeKeyNotification object:nil];

    _inspectorWKWebViewDelegate = delegate;

    if (!!_inspectorWKWebViewDelegate)
        [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(_handleWindowDidBecomeKey:) name:NSWindowDidBecomeKeyNotification object:nil];
}

- (IBAction)reload:(id)sender
{
    [self.inspectorWKWebViewDelegate inspectorWKWebViewReload:self];
}

- (IBAction)reloadFromOrigin:(id)sender
{
    [self.inspectorWKWebViewDelegate inspectorWKWebViewReloadFromOrigin:self];
}

- (void)viewWillMoveToWindow:(NSWindow *)newWindow
{
    [super viewWillMoveToWindow:newWindow];
    [self.inspectorWKWebViewDelegate inspectorWKWebView:self willMoveToWindow:newWindow];
}

- (void)viewDidMoveToWindow
{
    [super viewDidMoveToWindow];
    [self.inspectorWKWebViewDelegate inspectorWKWebViewDidMoveToWindow:self];
}

- (BOOL)becomeFirstResponder
{
    BOOL result = [super becomeFirstResponder];
    [self.inspectorWKWebViewDelegate inspectorWKWebViewDidBecomeActive:self];
    return result;
}

- (void)_handleWindowDidBecomeKey:(NSNotification *)notification
{
    if (notification.object == self.window)
        [self.inspectorWKWebViewDelegate inspectorWKWebViewDidBecomeActive:self];
}

@end

#endif

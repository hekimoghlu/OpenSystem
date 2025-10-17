/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#import "WebNodeHighlighter.h"

#import "DOMNodeInternal.h"
#import "WebFrameView.h"
#import "WebNodeHighlight.h"
#import "WebNodeHighlightView.h"
#import "WebViewInternal.h"
#import <WebCore/Page.h>

@implementation WebNodeHighlighter

- (id)initWithInspectedWebView:(WebView *)webView
{
    if (!(self = [super init]))
        return nil;

    // Don't retain to avoid a circular reference.
    _inspectedWebView = webView;

    return self;
}

- (void)dealloc
{
    ASSERT(!_currentHighlight);
    [super dealloc];
}

// MARK: -

- (void)highlight
{
#if !PLATFORM(IOS_FAMILY)
    // The scrollview's content view stays around between page navigations, so target it.
    NSView *view = [[[[[_inspectedWebView mainFrame] frameView] documentView] enclosingScrollView] contentView];
#else
    NSView *view = _inspectedWebView;
#endif
    if (![view window])
        return; // Skip the highlight if we have no window (e.g. hidden tab).
    
    if (!_currentHighlight) {
        _currentHighlight = [[WebNodeHighlight alloc] initWithTargetView:view inspectorController:&[_inspectedWebView page]->inspectorController()];
        [_currentHighlight setDelegate:self];
        [_currentHighlight attach];
    } else {
#if !PLATFORM(IOS_FAMILY)
        [[_currentHighlight highlightView] setNeedsDisplay:YES];
#else
        [_currentHighlight setNeedsDisplay];
#endif
    }
}

- (void)hideHighlight
{
    [_currentHighlight detach];
    [_currentHighlight setDelegate:nil];
    [_currentHighlight release];
    _currentHighlight = nil;
}

// MARK: -
// MARK: WebNodeHighlight delegate

- (void)didAttachWebNodeHighlight:(WebNodeHighlight *)highlight
{
    [_inspectedWebView setCurrentNodeHighlight:highlight];
}

- (void)willDetachWebNodeHighlight:(WebNodeHighlight *)highlight
{
    [_inspectedWebView setCurrentNodeHighlight:nil];
}
    
@end

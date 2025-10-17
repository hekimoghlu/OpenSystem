/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#import "WebInspector.h"

#import "WebFrameInternal.h"
#import "WebInspectorPrivate.h"
#import "WebInspectorFrontend.h"

#import <WebCore/Document.h>
#import <WebCore/InspectorController.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/Page.h>

NSString *WebInspectorDidStartSearchingForNode = @"WebInspectorDidStartSearchingForNode";
NSString *WebInspectorDidStopSearchingForNode = @"WebInspectorDidStopSearchingForNode";

@implementation WebInspector
- (id)initWithInspectedWebView:(WebView *)inspectedWebView
{
    if (!(self = [super init]))
        return nil;
    _inspectedWebView = inspectedWebView; // not retained to prevent a cycle

    return self;
}

- (void)dealloc
{
    [_frontend release];
    [super dealloc];
}

- (void)inspectedWebViewClosed
{
    [self close:nil];
    _inspectedWebView = nil;
}

- (void)showWindow
{
    if (auto* inspectedPage = core(_inspectedWebView))
        inspectedPage->inspectorController().show();
}

- (void)show:(id)sender
{
    [self showWindow];
}

- (void)showConsole:(id)sender
{
    [self showWindow];
    [_frontend showConsole];
}

- (BOOL)isDebuggingJavaScript
{
    return _frontend && [_frontend isDebuggingEnabled];
}

- (void)toggleDebuggingJavaScript:(id)sender
{
    [self showWindow];

    if ([self isDebuggingJavaScript])
        [_frontend setDebuggingEnabled:false];
    else
        [_frontend setDebuggingEnabled:true];
}

- (void)startDebuggingJavaScript:(id)sender
{
    if (_frontend)
        [_frontend setDebuggingEnabled:true];
}

- (void)stopDebuggingJavaScript:(id)sender
{
    if (_frontend)
        [_frontend setDebuggingEnabled:false];
}

- (BOOL)isProfilingJavaScript
{
    // No longer supported.
    return NO;
}

- (void)toggleProfilingJavaScript:(id)sender
{
    // No longer supported.
}

- (void)startProfilingJavaScript:(id)sender
{
    // No longer supported.
}

- (void)stopProfilingJavaScript:(id)sender
{
    // No longer supported.
}

- (BOOL)isJavaScriptProfilingEnabled
{
    // No longer supported.
    return NO;
}

- (void)setJavaScriptProfilingEnabled:(BOOL)enabled
{
    // No longer supported.
}

- (BOOL)isTimelineProfilingEnabled
{
    return _frontend && [_frontend isTimelineProfilingEnabled];
}

- (void)setTimelineProfilingEnabled:(BOOL)enabled
{
    if (_frontend)
        [_frontend setTimelineProfilingEnabled:enabled];
}

- (BOOL)isOpen
{
    return !!_frontend;
}

- (void)close:(id)sender 
{
    [_frontend close];
}

- (void)attach:(id)sender
{
    [_frontend attach];
}

- (void)detach:(id)sender
{
    [_frontend detach];
}

- (void)evaluateInFrontend:(id)sender script:(NSString *)script
{
    if (auto* inspectedPage = core(_inspectedWebView))
        inspectedPage->inspectorController().evaluateForTestInFrontend(script);
}

- (void)setFrontend:(WebInspectorFrontend *)frontend
{
    _frontend = [frontend retain];
}

- (void)releaseFrontend
{
    [_frontend release];
    _frontend = nil;
}
@end

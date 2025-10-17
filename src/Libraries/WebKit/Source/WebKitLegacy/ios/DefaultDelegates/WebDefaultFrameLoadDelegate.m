/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#if PLATFORM(IOS_FAMILY)

#import "WebDefaultFrameLoadDelegate.h"

#import <WebKitLegacy/WebFrameLoadDelegatePrivate.h>
#import "WebViewPrivate.h"

@implementation WebDefaultFrameLoadDelegate

+ (WebDefaultFrameLoadDelegate *)sharedFrameLoadDelegate
{
    static WebDefaultFrameLoadDelegate *sharedDelegate = nil;
    if (!sharedDelegate)
        sharedDelegate = [[WebDefaultFrameLoadDelegate alloc] init];
    return sharedDelegate;
}

- (void)webView:(WebView *)sender didStartProvisionalLoadForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didReceiveServerRedirectForProvisionalLoadForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didFailProvisionalLoadWithError:(NSError *)error forFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didCommitLoadForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didReceiveTitle:(NSString *)title forFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didFinishLoadForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didFailLoadWithError:(NSError *)error forFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didChangeLocationWithinPageForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender willPerformClientRedirectToURL:(NSURL *)URL delay:(NSTimeInterval)seconds fireDate:(NSDate *)date forFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didCancelClientRedirectForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender willCloseFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)webView didClearWindowObject:(WebScriptObject *)windowObject forFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)webView windowScriptObjectAvailable:(WebScriptObject *)windowScriptObject
{
}

- (void)webView:(WebView *)webView didCreateJavaScriptContext:(JSContext *)context forFrame:(WebFrame *)frame
{
}

- (void)webViewDidDisplayInsecureContent:(WebView *)webView
{
}

- (void)webView:(WebView *)webView didRunInsecureContent:(WebSecurityOrigin *)origin
{
}

- (void)webView:(WebView *)webView didDetectXSS:(NSURL *)insecureURL
{
}

- (void)webView:(WebView *)webView didClearWindowObjectForFrame:(WebFrame *)frame inScriptWorld:(WebScriptWorld *)world
{
}

- (void)webView:(WebView *)webView didPushStateWithinPageForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)webView didReplaceStateWithinPageForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)webView didPopStateWithinPageForFrame:(WebFrame *)frame
{
}

#pragma mark -
#pragma mark SPI defined in a category in WebViewPrivate.h

- (void)webView:(WebView *)sender didFirstLayoutInFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didFinishDocumentLoadForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didHandleOnloadEventsForFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didFirstVisuallyNonEmptyLayoutInFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)webView didClearInspectorWindowObject:(WebScriptObject *)windowObject forFrame:(WebFrame *)frame
{
}

- (void)webView:(WebView *)webView didRemoveFrameFromHierarchy:(WebFrame *)frame
{
}

- (void)webView:(WebView *)sender didLayout:(WebLayoutMilestones)milestones
{
}

@end

#endif // PLATFORM(IOS_FAMILY)

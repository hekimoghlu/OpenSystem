/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import "_WKInspectorInternal.h"
#import "_WKInspectorPrivateForTesting.h"

// This file exists to centralize all fragile code that is used by _WKInspector API tests. The tests
// trigger WebInspectorUI behavior by evaluating JavaScript or by calling internal methods.

static NSString *JavaScriptSnippetToOpenURLExternally(NSURL *url)
{
    return [NSString stringWithFormat:@"InspectorFrontendHost.openURLExternally(\"%@\")", url.absoluteString];
}

static NSString *JavaScriptSnippetToFetchURL(NSURL *url)
{
    return [NSString stringWithFormat:@"fetch(\"%@\")", url.absoluteString];
}

@implementation _WKInspector (WKTesting)

- (WKWebView *)inspectorWebView
{
    RefPtr page = _inspector->protectedInspectorPage();
    return page ? page->cocoaView().autorelease() : nil;
}

- (void)_fetchURLForTesting:(NSURL *)url
{
    _inspector->evaluateInFrontendForTesting(JavaScriptSnippetToFetchURL(url));
}

- (void)_openURLExternallyForTesting:(NSURL *)url useFrontendAPI:(BOOL)useFrontendAPI
{
    if (useFrontendAPI)
        _inspector->evaluateInFrontendForTesting(JavaScriptSnippetToOpenURLExternally(url));
    else {
        // Force the navigation request to be handled naturally through the
        // internal NavigationDelegate of WKInspectorViewController.
        [self.inspectorWebView loadRequest:[NSURLRequest requestWithURL:url]];
    }
}

@end

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#import "WebDefaultPolicyDelegate.h"

#import "WebDataSource.h"
#import "WebFrame.h"
#import "WebPolicyDelegatePrivate.h"
#import "WebViewInternal.h"
#import <Foundation/NSURLConnection.h>
#import <Foundation/NSURLRequest.h>
#import <Foundation/NSURLResponse.h>
#import <wtf/Assertions.h>
#import <wtf/EnumTraits.h>

@implementation WebDefaultPolicyDelegate

static WebDefaultPolicyDelegate *sharedDelegate = nil;

// Return a object with vanilla implementations of the protocol's methods
// Note this feature relies on our default delegate being stateless
+ (WebDefaultPolicyDelegate *)sharedPolicyDelegate
{
    if (!sharedDelegate) {
        sharedDelegate = [[WebDefaultPolicyDelegate alloc] init];
    }
    return sharedDelegate;
}

- (void)webView: (WebView *)wv unableToImplementPolicyWithError:(NSError *)error frame:(WebFrame *)frame
{
    LOG_ERROR("called unableToImplementPolicyWithError:%@ inFrame:%@", error, frame);
}


- (void)webView: (WebView *)wv decidePolicyForMIMEType:(NSString *)type
                                               request:(NSURLRequest *)request
                                                 frame:(WebFrame *)frame
                                      decisionListener:(id <WebPolicyDecisionListener>)listener
{
    if ([[request URL] isFileURL]) {
        BOOL isDirectory = NO;
        BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:[[request URL] path] isDirectory:&isDirectory];

        if (exists && !isDirectory && [wv _canShowMIMEType:type])
            [listener use];
        else
            [listener ignore];
    } else if ([wv _canShowMIMEType:type])
        [listener use];
    else
        [listener ignore];
}

- (void)webView: (WebView *)wv decidePolicyForNavigationAction:(NSDictionary *)actionInformation 
                                                       request:(NSURLRequest *)request
                                                         frame:(WebFrame *)frame
                                              decisionListener:(id <WebPolicyDecisionListener>)listener
{
    WebNavigationType navType = (WebNavigationType)[[actionInformation objectForKey:WebActionNavigationTypeKey] intValue];

    if ([WebView _canHandleRequest:request forMainFrame:frame == [wv mainFrame]]) {
        [listener use];
    } else if (enumToUnderlyingType(navType) == enumToUnderlyingType(WebNavigationTypePlugInRequest)) {
        [listener use];
    } else {
        // A file URL shouldn't fall through to here, but if it did,
        // it would be a security risk to open it.
        if (![[request URL] isFileURL]) {
#if !PLATFORM(IOS_FAMILY)
            [[NSWorkspace sharedWorkspace] openURL:[request URL]];
#endif
        }
        [listener ignore];
    }
}

- (void)webView: (WebView *)wv decidePolicyForNewWindowAction:(NSDictionary *)actionInformation 
                                                      request:(NSURLRequest *)request
                                                 newFrameName:(NSString *)frameName
                                             decisionListener:(id <WebPolicyDecisionListener>)listener
{
    [listener use];
}

// Temporary SPI needed for <rdar://problem/3951283> can view pages from the back/forward cache that should be disallowed by Parental Controls
- (BOOL)webView:(WebView *)webView shouldGoToHistoryItem:(WebHistoryItem *)item
{
    return YES;
}

@end


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

#import "WebDefaultResourceLoadDelegate.h"

#import <WebKitLegacy/WebResourceLoadDelegate.h>

@implementation WebDefaultResourceLoadDelegate

+ (WebDefaultResourceLoadDelegate *)sharedResourceLoadDelegate
{
    static WebDefaultResourceLoadDelegate *sharedDelegate = nil;
    if (!sharedDelegate)
        sharedDelegate = [[WebDefaultResourceLoadDelegate alloc] init];
    return sharedDelegate;
}

- (id)webView:(WebView *)sender identifierForInitialRequest:(NSURLRequest *)request fromDataSource:(WebDataSource *)dataSource
{
    return nil;
}

- (NSURLRequest *)webView:(WebView *)sender resource:(id)identifier willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse fromDataSource:(WebDataSource *)dataSource
{
    return request;
}

- (void)webView:(WebView *)sender resource:(id)identifier didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge fromDataSource:(WebDataSource *)dataSource
{
}

- (BOOL)webView:(WebView *)sender resource:(id)identifier canAuthenticateAgainstProtectionSpace:(NSURLProtectionSpace *)protectionSpace forDataSource:(WebDataSource *)dataSource
{
    return NO;
}

- (NSDictionary *)webView:(WebView *)sender connectionPropertiesForResource:(id)identifier dataSource:(WebDataSource *)dataSource
{
    return nil;
}

- (void)webView:(WebView *)sender resource:(id)identifier didReceiveResponse:(NSURLResponse *)response fromDataSource:(WebDataSource *)dataSource
{
}

- (void)webView:(WebView *)sender resource:(id)identifier didReceiveContentLength:(NSInteger)length fromDataSource:(WebDataSource *)dataSource
{
}

- (void)webView:(WebView *)sender resource:(id)identifier didFinishLoadingFromDataSource:(WebDataSource *)dataSource
{
}

- (void)webView:(WebView *)sender resource:(id)identifier didFailLoadingWithError:(NSError *)error fromDataSource:(WebDataSource *)dataSource
{
}

- (void)webView:(WebView *)sender plugInFailedWithError:(NSError *)error dataSource:(WebDataSource *)dataSource
{
}

- (void)webView:(WebView *)webView didLoadResourceFromMemoryCache:(NSURLRequest *)request response:(NSURLResponse *)response length:(NSInteger)length fromDataSource:(WebDataSource *)dataSource
{
}

- (BOOL)webView:(WebView *)webView resource:(id)identifier shouldUseCredentialStorageForDataSource:(WebDataSource *)dataSource
{
    return YES;
}

#pragma mark -
#pragma mark SPI defined in a category in WebViewPrivate.h

- (NSCachedURLResponse *)webView:(WebView *)sender resource:(id)identifier willCacheResponse:(NSCachedURLResponse *)response fromDataSource:(WebDataSource *)dataSource
{
    return response;
}

@end

#endif // PLATFORM(IOS_FAMILY)

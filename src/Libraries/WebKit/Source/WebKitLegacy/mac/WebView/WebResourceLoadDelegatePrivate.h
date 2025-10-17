/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

@class WebView;
@class WebDataSource;

@interface NSObject (WebResourceLoadDelegatePrivate)

- (void)webView:(WebView *)webView didLoadResourceFromMemoryCache:(NSURLRequest *)request response:(NSURLResponse *)response length:(NSInteger)length fromDataSource:(WebDataSource *)dataSource;
- (BOOL)webView:(WebView *)webView resource:(id)identifier shouldUseCredentialStorageForDataSource:(WebDataSource *)dataSource;

/*!
 @method webView:resource:canAuthenticateAgainstProtectionSpace:forDataSource:
 @abstract Inspect an NSURLProtectionSpace before an authentication attempt is made. Only used on Snow Leopard or newer.
 @param protectionSpace an NSURLProtectionSpace that will be used to generate an authentication challenge
 @result Return YES if the resource load delegate is prepared to respond to an authentication challenge generated with protectionSpace, NO otherwise
 */
- (BOOL)webView:(WebView *)sender resource:(id)identifier canAuthenticateAgainstProtectionSpace:(NSURLProtectionSpace *)protectionSpace forDataSource:(WebDataSource *)dataSource WEBKIT_AVAILABLE_MAC(10_6);

/*!
 @method webView:shouldPaintBrokenImageForURL:(NSURL*)imageURL
 @abstract This message is sent when an image cannot be decoded or displayed.
 @param imageURL The url of the broken image.
 @result return YES if WebKit should paint the default broken image.
 */
- (BOOL)webView:(WebView*)sender shouldPaintBrokenImageForURL:(NSURL*)imageURL;

#if TARGET_OS_IPHONE
- (id)webThreadWebView:(WebView *)sender identifierForInitialRequest:(NSURLRequest *)request fromDataSource:(WebDataSource *)dataSource;
- (NSURLRequest *)webThreadWebView:(WebView *)sender resource:(id)identifier willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse fromDataSource:(WebDataSource *)dataSource;
- (void)webThreadWebView:(WebView *)sender resource:(id)identifier didReceiveContentLength:(NSInteger)length fromDataSource:(WebDataSource *)dataSource;
- (void)webThreadWebView:(WebView *)sender resource:(id)identifier didReceiveResponse:(NSURLResponse *)response fromDataSource:(WebDataSource *)dataSource;
- (void)webThreadWebView:(WebView *)webView didLoadResourceFromMemoryCache:(NSURLRequest *)request response:(NSURLResponse *)response length:(NSInteger)length fromDataSource:(WebDataSource *)dataSource;
- (void)webThreadWebView:(WebView *)sender resource:(id)identifier didFinishLoadingFromDataSource:(WebDataSource *)dataSource;
- (void)webThreadWebView:(WebView *)sender resource:(id)identifier didFailLoadingWithError:(NSError *)error fromDataSource:(WebDataSource *)dataSource;
- (NSCachedURLResponse *)webThreadWebView:(WebView *)sender resource:(id)identifier willCacheResponse:(NSCachedURLResponse *)response fromDataSource:(WebDataSource *)dataSource;

/*!
 @method webView:connectionPropertiesForResource:
 @abstract Provide a CFStream level properties dictionary to be used by a connection
 @result return an NSDictionary with the desired stream properties or nil
 */
- (NSDictionary *)webView:(WebView *)sender connectionPropertiesForResource:(id)identifier dataSource:(WebDataSource *)dataSource;
#endif

@end

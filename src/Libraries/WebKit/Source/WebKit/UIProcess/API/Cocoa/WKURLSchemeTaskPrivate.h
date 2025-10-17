/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#import <WebKit/WKURLSchemeTask.h>

@class WKFrameInfo;

WK_API_AVAILABLE(macos(10.13), ios(11.0))
@protocol WKURLSchemeTaskPrivate <WKURLSchemeTask>

/*! @abstract Indicate the task was redirected.
 @param response The response causing a redirect to a new URL.
 @param newRequest The new proposed request to handle the redirection.
 @param completionHandler A completion handler called with the request you should use to commit to the redirection.
 @discussion When performing a load of a URL over a network, the server might decide a different URL should be loaded instead.
 A common example is an HTTP redirect.
 When this happens, you should notify WebKit by sending the servers response and a proposed new request to the WKURLSchemeTask.
 WebKit might decide that changes need to be make to the proposed request.
 This is communicated through the completionHandler which tells you the request you should make to commit to the redirection.

 An exception will be thrown if you make any other callbacks to the WKURLSchemeTask while this completionHandler is pending, other than didFailWithError:.
 An exception will be thrown if your app has been told to stop loading this task via the registered WKURLSchemeHandler object.
 */
- (void)_willPerformRedirection:(NSURLResponse *)response newRequest:(NSURLRequest *)request completionHandler:(void (^)(NSURLRequest *))completionHandler WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (void)_didPerformRedirection:(NSURLResponse *)response newRequest:(NSURLRequest *)request;

@property (nonatomic, readonly) BOOL _requestOnlyIfCached WK_API_AVAILABLE(macos(10.15), ios(13.0));
@property (nonatomic, readonly) WKFrameInfo *_frame WK_API_AVAILABLE(macos(11.0), ios(14.0));

@end

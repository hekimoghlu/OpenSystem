/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class WKWebView;
@protocol WKURLSchemeTask;

/*! A class conforming to the WKURLSchemeHandler protocol provides methods for
 loading resources with URL schemes that WebKit doesn't know how to handle itself.
 */
WK_SWIFT_UI_ACTOR
WK_API_AVAILABLE(macos(10.13), ios(11.0))
@protocol WKURLSchemeHandler <NSObject>

/*! @abstract Notifies your app to start loading the data for a particular resource 
 represented by the URL scheme handler task.
 @param webView The web view invoking the method.
 @param urlSchemeTask The task that your app should start loading data for.
 */
- (void)webView:(WKWebView *)webView startURLSchemeTask:(id <WKURLSchemeTask>)urlSchemeTask;

/*! @abstract Notifies your app to stop handling a URL scheme handler task.
 @param webView The web view invoking the method.
 @param urlSchemeTask The task that your app should stop handling.
 @discussion After your app is told to stop loading data for a URL scheme handler task
 it must not perform any callbacks for that task.
 An exception will be thrown if any callbacks are made on the URL scheme handler task
 after your app has been told to stop loading for it.
 */
- (void)webView:(WKWebView *)webView stopURLSchemeTask:(id <WKURLSchemeTask>)urlSchemeTask;

@end

NS_ASSUME_NONNULL_END

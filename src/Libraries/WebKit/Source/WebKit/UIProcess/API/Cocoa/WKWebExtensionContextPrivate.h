/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#import <WebKit/WKWebExtensionContext.h>

@class _WKWebExtensionSidebar;

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

@interface WKWebExtensionContext ()

/*! @abstract The extension background view used for the extension, or `nil` if the extension does not have background content or it is currently unloaded. */
@property (nonatomic, nullable, readonly) WKWebView *_backgroundWebView;

/*! @abstract The extension background content URL for the extension, or `nil` if the extension does not have background content. */
@property (nonatomic, nullable, readonly) NSURL *_backgroundContentURL;

/*!
 @abstract Sends a message to the JavaScript `browser.test.onMessage` API.
 @discussion Allows code to trigger a `browser.test.onMessage` event, enabling bidirectional communication during testing.
 @param message The message string to send.
 @param argument The optional JSON-serializable argument to include with the message. Must be JSON-serializable according to \c NSJSONSerialization.
 */
- (void)_sendTestMessage:(NSString *)message withArgument:(nullable id)argument;

/*!
 @abstract Retrieves the extension sidebar for a given tab, or the default sidebar if `nil` is passed.
 @param tab The tab for which to retrieve the extension sidebar, or `nil` to get the default sidebar.
 @discussion The returned object represents the sidebar specific to the tab when provided; otherwise, it returns the default sidebar.
 The default sidebar should not be directly displayed. When possible, specify the tab to get the most context-relevant sidebar.
 */
- (nullable _WKWebExtensionSidebar *)sidebarForTab:(nullable id <WKWebExtensionTab>)tab NS_SWIFT_NAME(sidebar(for:));

@end

WK_HEADER_AUDIT_END(nullability, sendability)

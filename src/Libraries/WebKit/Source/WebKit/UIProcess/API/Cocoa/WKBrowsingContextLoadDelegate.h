/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#import <WebKit/WKBrowsingContextController.h>

@class WKBackForwardListItem;

WK_CLASS_DEPRECATED_WITH_REPLACEMENT("WKNavigationDelegate", macos(10.10, 10.14.4), ios(8.0, 12.2))
@protocol WKBrowsingContextLoadDelegate <NSObject>
@optional

/* Sent when the provisional load begins. */
- (void)browsingContextControllerDidStartProvisionalLoad:(WKBrowsingContextController *)sender;

/* Sent if a server-side redirect was recieved. */
- (void)browsingContextControllerDidReceiveServerRedirectForProvisionalLoad:(WKBrowsingContextController *)sender;

/* Sent if the provisional load fails. */
- (void)browsingContextController:(WKBrowsingContextController *)sender didFailProvisionalLoadWithError:(NSError *)error;

/* Sent when the load gets committed. */
- (void)browsingContextControllerDidCommitLoad:(WKBrowsingContextController *)sender;

/* Sent when the load completes. */
- (void)browsingContextControllerDidFinishLoad:(WKBrowsingContextController *)sender;

/* Sent if the commited load fails. */
- (void)browsingContextController:(WKBrowsingContextController *)sender didFailLoadWithError:(NSError *)error;

- (void)browsingContextControllerDidStartProgress:(WKBrowsingContextController *)sender WK_API_DEPRECATED_WITH_REPLACEMENT("WKWebView.estimatedProgress", macos(10.10, 10.14.4), ios(8.0, 12.2));
- (void)browsingContextController:(WKBrowsingContextController *)sender estimatedProgressChangedTo:(double)estimatedProgress WK_API_DEPRECATED_WITH_REPLACEMENT("WKWebView.estimatedProgress", macos(10.10, 10.14.4), ios(8.0, 12.2));
- (void)browsingContextControllerDidFinishProgress:(WKBrowsingContextController *)sender WK_API_DEPRECATED_WITH_REPLACEMENT("WKWebView.estimatedProgress", macos(10.10, 10.14.4), ios(8.0, 12.2));

- (void)browsingContextControllerDidChangeBackForwardList:(WKBrowsingContextController *)sender addedItem:(WKBackForwardListItem *)addedItem removedItems:(NSArray *)removedItems WK_API_DEPRECATED_WITH_REPLACEMENT("WKWebView.backForwardList", macos(10.10, 10.14.4), ios(8.0, 12.2));

@end

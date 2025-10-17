/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#import <WebKit/WKNavigationData.h>

WK_CLASS_DEPRECATED_WITH_REPLACEMENT("WKBackForwardList", macos(10.10, 10.14.4), ios(8.0, 12.2))
@protocol WKBrowsingContextHistoryDelegate <NSObject>
@optional

- (void)browsingContextController:(WKBrowsingContextController *)browsingContextController didNavigateWithNavigationData:(WKNavigationData *)navigationData;
- (void)browsingContextController:(WKBrowsingContextController *)browsingContextController didPerformClientRedirectFromURL:(NSURL *)sourceURL toURL:(NSURL *)destinationURL;
- (void)browsingContextController:(WKBrowsingContextController *)browsingContextController didPerformServerRedirectFromURL:(NSURL *)sourceURL toURL:(NSURL *)destinationURL;
- (void)browsingContextController:(WKBrowsingContextController *)browsingContextController didUpdateHistoryTitle:(NSString *)title forURL:(NSURL *)URL;

@end

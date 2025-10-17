/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#import <WebKit/WKBrowsingContextGroup.h>
#import <WebKit/WKFoundation.h>
#import <WebKit/WKProcessGroup.h>

@class WKBackForwardList;
@class WKBackForwardListItem;
@protocol WKBrowsingContextHistoryDelegate;
@protocol WKBrowsingContextLoadDelegate;
@protocol WKBrowsingContextPolicyDelegate;

WK_CLASS_DEPRECATED_WITH_REPLACEMENT("WKWebView", macos(10.10, 10.14.4), ios(8.0, 12.2))
@interface WKBrowsingContextController : NSObject

#pragma mark Delegates

@property (weak) id <WKBrowsingContextLoadDelegate> loadDelegate;
@property (weak) id <WKBrowsingContextPolicyDelegate> policyDelegate;
@property (weak) id <WKBrowsingContextHistoryDelegate> historyDelegate;

#pragma mark Loading

+ (void)registerSchemeForCustomProtocol:(NSString *)scheme WK_API_DEPRECATED_WITH_REPLACEMENT("WKURLSchemeHandler", macos(10.10, 10.14.4), ios(8.0, 12.2));
+ (void)unregisterSchemeForCustomProtocol:(NSString *)scheme WK_API_DEPRECATED_WITH_REPLACEMENT("WKURLSchemeHandler", macos(10.10, 10.14.4), ios(8.0, 12.2));

/* Load a request. This is only valid for requests of non-file: URLs. Passing a
   file: URL will throw an exception. */
- (void)loadRequest:(NSURLRequest *)request;
- (void)loadRequest:(NSURLRequest *)request userData:(id)userData;

/* Load a file: URL. Opens the sandbox only for files within allowedDirectory.
    - Passing a non-file: URL to either parameter will yield an exception.
    - Passing nil as the allowedDirectory will open the entire file-system for
      reading.
*/
- (void)loadFileURL:(NSURL *)URL restrictToFilesWithin:(NSURL *)allowedDirectory;
- (void)loadFileURL:(NSURL *)URL restrictToFilesWithin:(NSURL *)allowedDirectory userData:(id)userData;

/* Load a webpage using the passed in string as its contents. */
- (void)loadHTMLString:(NSString *)HTMLString baseURL:(NSURL *)baseURL;
- (void)loadHTMLString:(NSString *)HTMLString baseURL:(NSURL *)baseURL userData:(id)userData;

- (void)loadAlternateHTMLString:(NSString *)string baseURL:(NSURL *)baseURL forUnreachableURL:(NSURL *)unreachableURL;

/* Load a webpage using the passed in data as its contents. */
- (void)loadData:(NSData *)data MIMEType:(NSString *)MIMEType textEncodingName:(NSString *)encodingName baseURL:(NSURL *)baseURL;
- (void)loadData:(NSData *)data MIMEType:(NSString *)MIMEType textEncodingName:(NSString *)encodingName baseURL:(NSURL *)baseURL userData:(id)userData;

/* Stops the load associated with the active URL. */
- (void)stopLoading;

/* Reload the currently active URL. */
- (void)reload;

/* Reload the currently active URL, bypassing caches. */
- (void)reloadFromOrigin;

@property (copy) NSString *applicationNameForUserAgent;
@property (copy) NSString *customUserAgent;

#pragma mark Back/Forward

/* Go to the next webpage in the back/forward list. */
- (void)goForward;

/* Whether there is a next webpage in the back/forward list. */
@property(readonly) BOOL canGoForward;

/* Go to the previous webpage in the back/forward list. */
- (void)goBack;

/* Whether there is a previous webpage in the back/forward list. */
@property(readonly) BOOL canGoBack;

- (void)goToBackForwardListItem:(WKBackForwardListItem *)item;

@property(readonly) WKBackForwardList *backForwardList;

#pragma mark Active Load Introspection

@property (readonly, getter=isLoading) BOOL loading;

/* URL for the active load. This is the URL that should be shown in user interface. */
@property(readonly) NSURL *activeURL;

/* URL for a request that has been sent, but no response has been received yet. */
@property(readonly) NSURL *provisionalURL;

/* URL for a request that has been received, and is now being used. */
@property(readonly) NSURL *committedURL;

@property(readonly) NSURL *unreachableURL;

@property(readonly) double estimatedProgress;

#pragma mark Active Document Introspection

/* Title of the document associated with the active load. */
@property(readonly) NSString *title;

@property (readonly) NSArray *certificateChain;

#pragma mark Zoom

/* Sets the text zoom for the active URL. */
@property CGFloat textZoom;

/* Sets the text zoom for the active URL. */
@property CGFloat pageZoom;

@end

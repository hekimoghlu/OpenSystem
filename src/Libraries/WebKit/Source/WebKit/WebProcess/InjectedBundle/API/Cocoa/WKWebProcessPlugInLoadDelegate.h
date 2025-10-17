/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#import <WebKit/_WKRenderingProgressEvents.h>
#import <WebKit/_WKSameDocumentNavigationType.h>

@class WKWebProcessPlugInBrowserContextController;
@class WKWebProcessPlugInFrame;
@class WKWebProcessPlugInScriptWorld;

@protocol WKWebProcessPlugInLoadDelegate <NSObject>
@optional

// Frame loading

- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didStartProvisionalLoadForFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didReceiveServerRedirectForProvisionalLoadForFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didCommitLoadForFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didFinishDocumentLoadForFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didFailProvisionalLoadWithErrorForFrame:(WKWebProcessPlugInFrame *)frame error:(NSError *)error;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didFailLoadWithErrorForFrame:(WKWebProcessPlugInFrame *)frame error:(NSError *)error;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didFinishLoadForFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didSameDocumentNavigation:(_WKSameDocumentNavigationType)navigationType forFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller didClearWindowObjectForFrame:(WKWebProcessPlugInFrame *)frame inScriptWorld:(WKWebProcessPlugInScriptWorld *)scriptWorld;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller globalObjectIsAvailableForFrame:(WKWebProcessPlugInFrame *)frame inScriptWorld:(WKWebProcessPlugInScriptWorld *)scriptWorld;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller serviceWorkerGlobalObjectIsAvailableForFrame:(WKWebProcessPlugInFrame *)frame inScriptWorld:(WKWebProcessPlugInScriptWorld *)scriptWorld;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller willInjectUserScriptForFrame:(WKWebProcessPlugInFrame *)frame inScriptWorld:(WKWebProcessPlugInScriptWorld *)scriptWorld;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller didRemoveFrameFromHierarchy:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller didHandleOnloadEventsForFrame:(WKWebProcessPlugInFrame *)frame;

// Layout
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didLayoutForFrame:(WKWebProcessPlugInFrame *)frame;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller renderingProgressDidChange:(_WKRenderingProgressEvents)events;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController*)controller didFirstVisuallyNonEmptyLayoutForFrame:(WKWebProcessPlugInFrame *)frame;
- (_WKRenderingProgressEvents)webProcessPlugInBrowserContextControllerRenderingProgressEvents:(WKWebProcessPlugInBrowserContextController *)controller;

// Resource loading

- (NSURLRequest *)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame willSendRequestForResource:(uint64_t)resource request:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse;
- (NSURLRequest *)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse;

- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame didInitiateLoadForResource:(uint64_t)resource request:(NSURLRequest *)request pageIsProvisionallyLoading:(BOOL)pageIsProvisionallyLoading;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame didInitiateLoadForResource:(uint64_t)resource request:(NSURLRequest *)request;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame didReceiveResponse:(NSURLResponse *)response forResource:(uint64_t)resource;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame didFinishLoadForResource:(uint64_t)resource;
- (void)webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller frame:(WKWebProcessPlugInFrame *)frame didFailLoadForResource:(uint64_t)resource error:(NSError *)error;

@end

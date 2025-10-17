/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#ifndef WKBundlePageLoaderClient_h
#define WKBundlePageLoaderClient_h

#include <WebKit/WKBase.h>
#include <WebKit/WKPageLoadTypes.h>

typedef void (*WKBundlePageDidStartProvisionalLoadForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKErrorRef error, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidCommitLoadForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidDocumentFinishLoadForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidFinishLoadForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidFinishDocumentLoadForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidFinishProgressCallback)(WKBundlePageRef page, const void *clientInfo);
typedef void (*WKBundlePageDidFailLoadWithErrorForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKErrorRef error, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidSameDocumentNavigationForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKSameDocumentNavigationType type, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidReceiveTitleForFrameCallback)(WKBundlePageRef page, WKStringRef title, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidRemoveFrameFromHierarchyCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidDisplayInsecureContentForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidRunInsecureContentForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidDetectXSSForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidFirstLayoutForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidLayoutForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, const void* clientInfo);
typedef void (*WKBundlePageDidClearWindowObjectForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKBundleScriptWorldRef world, const void *clientInfo);
typedef void (*WKBundlePageDidCancelClientRedirectForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, const void *clientInfo);
typedef void (*WKBundlePageWillPerformClientRedirectForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKURLRef url, double delay, double date, const void *clientInfo);
typedef void (*WKBundlePageDidHandleOnloadEventsForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef frame, const void *clientInfo);
typedef bool (*WKBundlePageShouldGoToBackForwardListItemCallback)(WKBundlePageRef page, WKBundleBackForwardListItemRef item, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageGlobalObjectIsAvailableForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef, WKBundleScriptWorldRef, const void* clientInfo);
typedef void (*WKBundlePageServiceWorkerGlobalObjectIsAvailableForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef, WKBundleScriptWorldRef, const void* clientInfo);
typedef void (*WKBundlePageWillInjectUserScriptForFrameCallback)(WKBundlePageRef page, WKBundleFrameRef, WKBundleScriptWorldRef, const void* clientInfo);
typedef void (*WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback)(WKBundlePageRef page, WKBundleDOMWindowExtensionRef, const void* clientInfo);
typedef void (*WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback)(WKBundlePageRef page, WKBundleDOMWindowExtensionRef, const void* clientInfo);
typedef void (*WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback)(WKBundlePageRef page, WKBundleDOMWindowExtensionRef, const void* clientInfo);
typedef bool (*WKBundlePageShouldForceUniversalAccessFromLocalURLCallback)(WKBundlePageRef, WKStringRef url, const void* clientInfo);
typedef void (*WKBundlePageDidLayoutCallback)(WKBundlePageRef page, WKLayoutMilestones milestones, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageFeaturesUsedInPageCallback)(WKBundlePageRef page, WKArrayRef featureStrings, const void *clientInfo);
typedef void (*WKBundlePageWillLoadURLRequestCallback)(WKBundlePageRef page, WKURLRequestRef request, WKTypeRef userData, const void *clientInfo);
typedef void (*WKBundlePageWillLoadDataRequestCallback)(WKBundlePageRef page, WKURLRequestRef request, WKDataRef data, WKStringRef MIMEType, WKStringRef encodingName, WKURLRef unreachableURL, WKTypeRef userData, const void *clientInfo);
typedef WKLayoutMilestones (*WKBundlePageLayoutMilestonesCallback)(const void* clientInfo);

typedef struct WKBundlePageLoaderClientBase {
    int                                                                     version;
    const void *                                                            clientInfo;
} WKBundlePageLoaderClientBase;

typedef struct WKBundlePageLoaderClientV0 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;
} WKBundlePageLoaderClientV0;

typedef struct WKBundlePageLoaderClientV1 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;
} WKBundlePageLoaderClientV1;

typedef struct WKBundlePageLoaderClientV2 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;
} WKBundlePageLoaderClientV2;

typedef struct WKBundlePageLoaderClientV3 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;
} WKBundlePageLoaderClientV3;

typedef struct WKBundlePageLoaderClientV4 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;
} WKBundlePageLoaderClientV4;

typedef struct WKBundlePageLoaderClientV5 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;

    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;
} WKBundlePageLoaderClientV5;

typedef struct WKBundlePageLoaderClientV6 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;

    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;

    // Version 6
    WKBundlePageWillLoadURLRequestCallback                                  willLoadURLRequest;
    WKBundlePageWillLoadDataRequestCallback                                 willLoadDataRequest;
} WKBundlePageLoaderClientV6;

typedef struct WKBundlePageLoaderClientV7 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;

    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;

    // Version 6
    WKBundlePageWillLoadURLRequestCallback                                  willLoadURLRequest;
    WKBundlePageWillLoadDataRequestCallback                                 willLoadDataRequest;

    // Version 7
    void *                                                                  willDestroyFrame_unavailable;
} WKBundlePageLoaderClientV7;

typedef struct WKBundlePageLoaderClientV8 {
    WKBundlePageLoaderClientBase                                            base;
    
    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;
    
    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;
    
    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;
    
    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;
    
    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;
    
    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;
    
    // Version 6
    WKBundlePageWillLoadURLRequestCallback                                  willLoadURLRequest;
    WKBundlePageWillLoadDataRequestCallback                                 willLoadDataRequest;
    
    // Version 7
    void *                                                                  willDestroyFrame_unavailable;
    
    // Version 8
    void*                                                                   userAgentForURL_unavailable;
} WKBundlePageLoaderClientV8;

typedef struct WKBundlePageLoaderClientV9 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;

    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;

    // Version 6
    WKBundlePageWillLoadURLRequestCallback                                  willLoadURLRequest;
    WKBundlePageWillLoadDataRequestCallback                                 willLoadDataRequest;

    // Version 7
    void *                                                                  willDestroyFrame_unavailable;

    // Version 8
    void*                                                                   userAgentForURL_unavailable;

    // Version 9
    WKBundlePageWillInjectUserScriptForFrameCallback                        willInjectUserScriptForFrame;
} WKBundlePageLoaderClientV9;

typedef struct WKBundlePageLoaderClientV10 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;

    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;

    // Version 6
    WKBundlePageWillLoadURLRequestCallback                                  willLoadURLRequest;
    WKBundlePageWillLoadDataRequestCallback                                 willLoadDataRequest;

    // Version 7
    void *                                                                  willDestroyFrame_unavailable;

    // Version 8
    void*                                                                   userAgentForURL_unavailable;

    // Version 9
    WKBundlePageWillInjectUserScriptForFrameCallback                        willInjectUserScriptForFrame;

    // Version 10
    WKBundlePageLayoutMilestonesCallback                                    layoutMilestones;
} WKBundlePageLoaderClientV10;

typedef struct WKBundlePageLoaderClientV11 {
    WKBundlePageLoaderClientBase                                            base;

    // Version 0.
    WKBundlePageDidStartProvisionalLoadForFrameCallback                     didStartProvisionalLoadForFrame;
    WKBundlePageDidReceiveServerRedirectForProvisionalLoadForFrameCallback  didReceiveServerRedirectForProvisionalLoadForFrame;
    WKBundlePageDidFailProvisionalLoadWithErrorForFrameCallback             didFailProvisionalLoadWithErrorForFrame;
    WKBundlePageDidCommitLoadForFrameCallback                               didCommitLoadForFrame;
    WKBundlePageDidFinishDocumentLoadForFrameCallback                       didFinishDocumentLoadForFrame;
    WKBundlePageDidFinishLoadForFrameCallback                               didFinishLoadForFrame;
    WKBundlePageDidFailLoadWithErrorForFrameCallback                        didFailLoadWithErrorForFrame;
    WKBundlePageDidSameDocumentNavigationForFrameCallback                   didSameDocumentNavigationForFrame;
    WKBundlePageDidReceiveTitleForFrameCallback                             didReceiveTitleForFrame;
    WKBundlePageDidFirstLayoutForFrameCallback                              didFirstLayoutForFrame;
    WKBundlePageDidFirstVisuallyNonEmptyLayoutForFrameCallback              didFirstVisuallyNonEmptyLayoutForFrame;
    WKBundlePageDidRemoveFrameFromHierarchyCallback                         didRemoveFrameFromHierarchy;
    WKBundlePageDidDisplayInsecureContentForFrameCallback                   didDisplayInsecureContentForFrame;
    WKBundlePageDidRunInsecureContentForFrameCallback                       didRunInsecureContentForFrame;
    WKBundlePageDidClearWindowObjectForFrameCallback                        didClearWindowObjectForFrame;
    WKBundlePageDidCancelClientRedirectForFrameCallback                     didCancelClientRedirectForFrame;
    WKBundlePageWillPerformClientRedirectForFrameCallback                   willPerformClientRedirectForFrame;
    WKBundlePageDidHandleOnloadEventsForFrameCallback                       didHandleOnloadEventsForFrame;

    // Version 1.
    WKBundlePageDidLayoutForFrameCallback                                   didLayoutForFrame;
    void *                                                                  didNewFirstVisuallyNonEmptyLayout_unavailable;
    WKBundlePageDidDetectXSSForFrameCallback                                didDetectXSSForFrame;
    WKBundlePageShouldGoToBackForwardListItemCallback                       shouldGoToBackForwardListItem;
    WKBundlePageGlobalObjectIsAvailableForFrameCallback                     globalObjectIsAvailableForFrame;
    WKBundlePageWillDisconnectDOMWindowExtensionFromGlobalObjectCallback    willDisconnectDOMWindowExtensionFromGlobalObject;
    WKBundlePageDidReconnectDOMWindowExtensionToGlobalObjectCallback        didReconnectDOMWindowExtensionToGlobalObject;
    WKBundlePageWillDestroyGlobalObjectForDOMWindowExtensionCallback        willDestroyGlobalObjectForDOMWindowExtension;

    // Version 2
    WKBundlePageDidFinishProgressCallback                                   didFinishProgress;
    WKBundlePageShouldForceUniversalAccessFromLocalURLCallback              shouldForceUniversalAccessFromLocalURL;

    // Version 3
    void *                                                                  didReceiveIntentForFrame_unavailable;
    void *                                                                  registerIntentServiceForFrame_unavailable;

    // Version 4
    WKBundlePageDidLayoutCallback                                           didLayout;

    // Version 5
    WKBundlePageFeaturesUsedInPageCallback                                  featuresUsedInPage;

    // Version 6
    WKBundlePageWillLoadURLRequestCallback                                  willLoadURLRequest;
    WKBundlePageWillLoadDataRequestCallback                                 willLoadDataRequest;

    // Version 7
    void *                                                                  willDestroyFrame_unavailable;

    // Version 8
    void*                                                                   userAgentForURL_unavailable;

    // Version 9
    WKBundlePageWillInjectUserScriptForFrameCallback                        willInjectUserScriptForFrame;

    // Version 10
    WKBundlePageLayoutMilestonesCallback                                    layoutMilestones;

    // Version 11
    WKBundlePageServiceWorkerGlobalObjectIsAvailableForFrameCallback        serviceWorkerGlobalObjectIsAvailableForFrame;
} WKBundlePageLoaderClientV11;

#endif // WKBundlePageLoaderClient_h

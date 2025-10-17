/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#ifndef WKPageNavigationClient_h
#define WKPageNavigationClient_h

#include <WebKit/WKBase.h>
#include <WebKit/WKPageLoadTypes.h>
#include <WebKit/WKPageRenderingProgressEvents.h>
#include <WebKit/WKPluginLoadPolicy.h>
#include <WebKit/WKProcessTerminationReason.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKPageNavigationDecidePolicyForNavigationActionCallback)(WKPageRef page, WKNavigationActionRef navigationAction, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDecidePolicyForNavigationResponseCallback)(WKPageRef page, WKNavigationResponseRef navigationResponse, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidStartProvisionalNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidReceiveServerRedirectForProvisionalNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidFailProvisionalNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKErrorRef error, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidCommitNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidFinishNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidFailNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKErrorRef error, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidFailProvisionalLoadInSubframeCallback)(WKPageRef page, WKNavigationRef navigation, WKFrameInfoRef subframe, WKErrorRef error, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidFinishDocumentLoadCallback)(WKPageRef page, WKNavigationRef navigation, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationDidSameDocumentNavigationCallback)(WKPageRef page, WKNavigationRef navigation, WKSameDocumentNavigationType navigationType, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageNavigationRenderingProgressDidChangeCallback)(WKPageRef page, WKPageRenderingProgressEvents progressEvents, WKTypeRef userData, const void* clientInfo);
    
typedef bool (*WKPageNavigationCanAuthenticateAgainstProtectionSpaceCallback)(WKPageRef page, WKProtectionSpaceRef protectionSpace, const void* clientInfo);

typedef void (*WKPageNavigationDidReceiveAuthenticationChallengeCallback)(WKPageRef page, WKAuthenticationChallengeRef challenge, const void* clientInfo);

typedef void (*WKPageNavigationWebProcessDidCrashCallback)(WKPageRef page, const void* clientInfo);

typedef void (*WKPageNavigationWebProcessDidTerminateCallback)(WKPageRef page, WKProcessTerminationReason reason, const void* clientInfo);

typedef WKDataRef (*WKPageNavigationCopyWebCryptoMasterKeyCallback)(WKPageRef page, const void* clientInfo);

typedef WKStringRef (*WKPageNavigationCopySignedPublicKeyAndChallengeStringCallback)(WKPageRef page, const void* clientInfo);

typedef void (*WKPageNavigationActionDidBecomeDownloadCallback)(WKPageRef page, WKNavigationActionRef navigationAction, WKDownloadRef download, const void* clientInfo);

typedef void (*WKPageNavigationResponseDidBecomeDownloadCallback)(WKPageRef page, WKNavigationResponseRef navigationResponse, WKDownloadRef download, const void* clientInfo);

typedef void (*WKPageNavigationContextMenuDidCreateDownloadCallback)(WKPageRef page, WKDownloadRef download, const void* clientInfo);

typedef WKPluginLoadPolicy (*WKPageNavigationDecidePolicyForPluginLoadCallback)(WKPageRef page, WKPluginLoadPolicy currentPluginLoadPolicy, WKDictionaryRef pluginInfoDictionary, WKStringRef* unavailabilityDescription, const void* clientInfo);

typedef void (*WKPageNavigationDidBeginNavigationGesture)(WKPageRef page, const void* clientInfo);

typedef void (*WKPageNavigationWillEndNavigationGesture)(WKPageRef page, WKBackForwardListItemRef backForwardListItem, const void* clientInfo);

typedef void (*WKPageNavigationDidEndNavigationGesture)(WKPageRef page, WKBackForwardListItemRef backForwardListItem, const void* clientInfo);

typedef void (*WKPageNavigationDidRemoveNavigationGestureSnapshot)(WKPageRef page, const void* clientInfo);

typedef void (*WKPageNavigationContentRuleListNotificationCallback)(WKPageRef, WKURLRef, WKArrayRef, WKArrayRef, const void* clientInfo);

typedef struct WKPageNavigationClientBase {
    int version;
    const void* clientInfo;
} WKPageNavigationClientBase;

typedef struct WKPageNavigationClientV0 {
    WKPageNavigationClientBase base;

    // Version 0.
    WKPageNavigationDecidePolicyForNavigationActionCallback decidePolicyForNavigationAction;
    WKPageNavigationDecidePolicyForNavigationResponseCallback decidePolicyForNavigationResponse;
    WKPageNavigationDecidePolicyForPluginLoadCallback decidePolicyForPluginLoad;
    WKPageNavigationDidStartProvisionalNavigationCallback didStartProvisionalNavigation;
    WKPageNavigationDidReceiveServerRedirectForProvisionalNavigationCallback didReceiveServerRedirectForProvisionalNavigation;
    WKPageNavigationDidFailProvisionalNavigationCallback didFailProvisionalNavigation;
    WKPageNavigationDidCommitNavigationCallback didCommitNavigation;
    WKPageNavigationDidFinishNavigationCallback didFinishNavigation;
    WKPageNavigationDidFailNavigationCallback didFailNavigation;
    WKPageNavigationDidFailProvisionalLoadInSubframeCallback didFailProvisionalLoadInSubframe;
    WKPageNavigationDidFinishDocumentLoadCallback didFinishDocumentLoad;
    WKPageNavigationDidSameDocumentNavigationCallback didSameDocumentNavigation;
    WKPageNavigationRenderingProgressDidChangeCallback renderingProgressDidChange;
    WKPageNavigationCanAuthenticateAgainstProtectionSpaceCallback canAuthenticateAgainstProtectionSpace;
    WKPageNavigationDidReceiveAuthenticationChallengeCallback didReceiveAuthenticationChallenge;
    WKPageNavigationWebProcessDidCrashCallback webProcessDidCrash;
    WKPageNavigationCopyWebCryptoMasterKeyCallback copyWebCryptoMasterKey;
    WKPageNavigationDidBeginNavigationGesture didBeginNavigationGesture;
    WKPageNavigationWillEndNavigationGesture willEndNavigationGesture;
    WKPageNavigationDidEndNavigationGesture didEndNavigationGesture;
    WKPageNavigationDidRemoveNavigationGestureSnapshot didRemoveNavigationGestureSnapshot;
} WKPageNavigationClientV0;

typedef struct WKPageNavigationClientV1 {
    WKPageNavigationClientBase base;

    // Version 0.
    WKPageNavigationDecidePolicyForNavigationActionCallback decidePolicyForNavigationAction;
    WKPageNavigationDecidePolicyForNavigationResponseCallback decidePolicyForNavigationResponse;
    WKPageNavigationDecidePolicyForPluginLoadCallback decidePolicyForPluginLoad;
    WKPageNavigationDidStartProvisionalNavigationCallback didStartProvisionalNavigation;
    WKPageNavigationDidReceiveServerRedirectForProvisionalNavigationCallback didReceiveServerRedirectForProvisionalNavigation;
    WKPageNavigationDidFailProvisionalNavigationCallback didFailProvisionalNavigation;
    WKPageNavigationDidCommitNavigationCallback didCommitNavigation;
    WKPageNavigationDidFinishNavigationCallback didFinishNavigation;
    WKPageNavigationDidFailNavigationCallback didFailNavigation;
    WKPageNavigationDidFailProvisionalLoadInSubframeCallback didFailProvisionalLoadInSubframe;
    WKPageNavigationDidFinishDocumentLoadCallback didFinishDocumentLoad;
    WKPageNavigationDidSameDocumentNavigationCallback didSameDocumentNavigation;
    WKPageNavigationRenderingProgressDidChangeCallback renderingProgressDidChange;
    WKPageNavigationCanAuthenticateAgainstProtectionSpaceCallback canAuthenticateAgainstProtectionSpace;
    WKPageNavigationDidReceiveAuthenticationChallengeCallback didReceiveAuthenticationChallenge;
    WKPageNavigationWebProcessDidCrashCallback webProcessDidCrash;
    WKPageNavigationCopyWebCryptoMasterKeyCallback copyWebCryptoMasterKey;
    WKPageNavigationDidBeginNavigationGesture didBeginNavigationGesture;
    WKPageNavigationWillEndNavigationGesture willEndNavigationGesture;
    WKPageNavigationDidEndNavigationGesture didEndNavigationGesture;
    WKPageNavigationDidRemoveNavigationGestureSnapshot didRemoveNavigationGestureSnapshot;

    // Version 1.
    WKPageNavigationWebProcessDidTerminateCallback webProcessDidTerminate;
} WKPageNavigationClientV1;

typedef struct WKPageNavigationClientV2 {
    WKPageNavigationClientBase base;
    
    // Version 0.
    WKPageNavigationDecidePolicyForNavigationActionCallback decidePolicyForNavigationAction;
    WKPageNavigationDecidePolicyForNavigationResponseCallback decidePolicyForNavigationResponse;
    WKPageNavigationDecidePolicyForPluginLoadCallback decidePolicyForPluginLoad;
    WKPageNavigationDidStartProvisionalNavigationCallback didStartProvisionalNavigation;
    WKPageNavigationDidReceiveServerRedirectForProvisionalNavigationCallback didReceiveServerRedirectForProvisionalNavigation;
    WKPageNavigationDidFailProvisionalNavigationCallback didFailProvisionalNavigation;
    WKPageNavigationDidCommitNavigationCallback didCommitNavigation;
    WKPageNavigationDidFinishNavigationCallback didFinishNavigation;
    WKPageNavigationDidFailNavigationCallback didFailNavigation;
    WKPageNavigationDidFailProvisionalLoadInSubframeCallback didFailProvisionalLoadInSubframe;
    WKPageNavigationDidFinishDocumentLoadCallback didFinishDocumentLoad;
    WKPageNavigationDidSameDocumentNavigationCallback didSameDocumentNavigation;
    WKPageNavigationRenderingProgressDidChangeCallback renderingProgressDidChange;
    WKPageNavigationCanAuthenticateAgainstProtectionSpaceCallback canAuthenticateAgainstProtectionSpace;
    WKPageNavigationDidReceiveAuthenticationChallengeCallback didReceiveAuthenticationChallenge;
    WKPageNavigationWebProcessDidCrashCallback webProcessDidCrash;
    WKPageNavigationCopyWebCryptoMasterKeyCallback copyWebCryptoMasterKey;
    WKPageNavigationDidBeginNavigationGesture didBeginNavigationGesture;
    WKPageNavigationWillEndNavigationGesture willEndNavigationGesture;
    WKPageNavigationDidEndNavigationGesture didEndNavigationGesture;
    WKPageNavigationDidRemoveNavigationGestureSnapshot didRemoveNavigationGestureSnapshot;
    
    // Version 1.
    WKPageNavigationWebProcessDidTerminateCallback webProcessDidTerminate;

    // Version 2.
    WKPageNavigationContentRuleListNotificationCallback contentRuleListNotification;
} WKPageNavigationClientV2;

typedef struct WKPageNavigationClientV3 {
    WKPageNavigationClientBase base;

    // Version 0.
    WKPageNavigationDecidePolicyForNavigationActionCallback decidePolicyForNavigationAction;
    WKPageNavigationDecidePolicyForNavigationResponseCallback decidePolicyForNavigationResponse;
    WKPageNavigationDecidePolicyForPluginLoadCallback decidePolicyForPluginLoad;
    WKPageNavigationDidStartProvisionalNavigationCallback didStartProvisionalNavigation;
    WKPageNavigationDidReceiveServerRedirectForProvisionalNavigationCallback didReceiveServerRedirectForProvisionalNavigation;
    WKPageNavigationDidFailProvisionalNavigationCallback didFailProvisionalNavigation;
    WKPageNavigationDidCommitNavigationCallback didCommitNavigation;
    WKPageNavigationDidFinishNavigationCallback didFinishNavigation;
    WKPageNavigationDidFailNavigationCallback didFailNavigation;
    WKPageNavigationDidFailProvisionalLoadInSubframeCallback didFailProvisionalLoadInSubframe;
    WKPageNavigationDidFinishDocumentLoadCallback didFinishDocumentLoad;
    WKPageNavigationDidSameDocumentNavigationCallback didSameDocumentNavigation;
    WKPageNavigationRenderingProgressDidChangeCallback renderingProgressDidChange;
    WKPageNavigationCanAuthenticateAgainstProtectionSpaceCallback canAuthenticateAgainstProtectionSpace;
    WKPageNavigationDidReceiveAuthenticationChallengeCallback didReceiveAuthenticationChallenge;
    WKPageNavigationWebProcessDidCrashCallback webProcessDidCrash;
    WKPageNavigationCopyWebCryptoMasterKeyCallback copyWebCryptoMasterKey;
    WKPageNavigationDidBeginNavigationGesture didBeginNavigationGesture;
    WKPageNavigationWillEndNavigationGesture willEndNavigationGesture;
    WKPageNavigationDidEndNavigationGesture didEndNavigationGesture;
    WKPageNavigationDidRemoveNavigationGestureSnapshot didRemoveNavigationGestureSnapshot;

    // Version 1.
    WKPageNavigationWebProcessDidTerminateCallback webProcessDidTerminate;

    // Version 2.
    WKPageNavigationContentRuleListNotificationCallback contentRuleListNotification;

    // Version 3.
    WKPageNavigationCopySignedPublicKeyAndChallengeStringCallback copySignedPublicKeyAndChallengeString;
    WKPageNavigationActionDidBecomeDownloadCallback navigationActionDidBecomeDownload;
    WKPageNavigationResponseDidBecomeDownloadCallback navigationResponseDidBecomeDownload;
    WKPageNavigationContextMenuDidCreateDownloadCallback contextMenuDidCreateDownload;
} WKPageNavigationClientV3;

#ifdef __cplusplus
}
#endif

#endif // WKPageNavigationClient_h

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
// This header contains WebView declarations that can be used anywhere in WebKit, but are neither SPI nor API.

#import <JavaScriptCore/JSBase.h>

#if PLATFORM(IOS_FAMILY)
#import <WebKitLegacy/WAKAppKitStubs.h>
#endif

@class WebView;

struct WebResourceDelegateImplementationCache {
    IMP didReceiveAuthenticationChallengeFunc;
#if USE(PROTECTION_SPACE_AUTH_CALLBACK)
    IMP canAuthenticateAgainstProtectionSpaceFunc;
#endif

#if PLATFORM(IOS_FAMILY)
    IMP connectionPropertiesFunc;
    IMP webThreadDidFinishLoadingFromDataSourceFunc;
    IMP webThreadDidFailLoadingWithErrorFromDataSourceFunc;
    IMP webThreadIdentifierForRequestFunc;
    IMP webThreadDidLoadResourceFromMemoryCacheFunc;
    IMP webThreadWillSendRequestFunc;
    IMP webThreadDidReceiveResponseFunc;
    IMP webThreadDidReceiveContentLengthFunc;
    IMP webThreadWillCacheResponseFunc;
#endif

    IMP identifierForRequestFunc;
    IMP willSendRequestFunc;
    IMP didReceiveResponseFunc;
    IMP didReceiveContentLengthFunc;
    IMP didFinishLoadingFromDataSourceFunc;
    IMP didFailLoadingWithErrorFromDataSourceFunc;
    IMP didLoadResourceFromMemoryCacheFunc;
    IMP willCacheResponseFunc;
    IMP plugInFailedWithErrorFunc;
    IMP shouldUseCredentialStorageFunc;
    IMP shouldPaintBrokenImageForURLFunc;
};

struct WebFrameLoadDelegateImplementationCache {
#if JSC_OBJC_API_ENABLED
    IMP didCreateJavaScriptContextForFrameFunc;
#endif
    IMP didClearWindowObjectForFrameFunc;
    IMP didClearWindowObjectForFrameInScriptWorldFunc;
    IMP didClearInspectorWindowObjectForFrameFunc;
    IMP windowScriptObjectAvailableFunc;
    IMP didHandleOnloadEventsForFrameFunc;
    IMP didReceiveServerRedirectForProvisionalLoadForFrameFunc;
    IMP didCancelClientRedirectForFrameFunc;
    IMP willPerformClientRedirectToURLDelayFireDateForFrameFunc;
    IMP didChangeLocationWithinPageForFrameFunc;
    IMP didPushStateWithinPageForFrameFunc;
    IMP didReplaceStateWithinPageForFrameFunc;
    IMP didPopStateWithinPageForFrameFunc;
    IMP willCloseFrameFunc;
    IMP didStartProvisionalLoadForFrameFunc;
    IMP didReceiveTitleForFrameFunc;
    IMP didCommitLoadForFrameFunc;
    IMP didFailProvisionalLoadWithErrorForFrameFunc;
    IMP didFailLoadWithErrorForFrameFunc;
    IMP didFinishLoadForFrameFunc;
    IMP didFirstLayoutInFrameFunc;
    IMP didFirstVisuallyNonEmptyLayoutInFrameFunc;
    IMP didLayoutFunc;
    IMP didReceiveIconForFrameFunc;
    IMP didFinishDocumentLoadForFrameFunc;
    IMP didDisplayInsecureContentFunc;
    IMP didRunInsecureContentFunc;
    IMP didDetectXSSFunc;
    IMP didRemoveFrameFromHierarchyFunc;
#if PLATFORM(IOS_FAMILY)
    IMP webThreadDidLayoutFunc;
#endif
};

struct WebScriptDebugDelegateImplementationCache {
    BOOL didParseSourceExpectsBaseLineNumber;
    BOOL exceptionWasRaisedExpectsHasHandlerFlag;
    IMP didParseSourceFunc;
    IMP failedToParseSourceFunc;
    IMP exceptionWasRaisedFunc;
};

struct WebHistoryDelegateImplementationCache {
    IMP navigatedFunc;
    IMP clientRedirectFunc;
    IMP serverRedirectFunc;
    IMP deprecatedSetTitleFunc;
    IMP setTitleFunc;
    IMP populateVisitedLinksFunc;
};

WebResourceDelegateImplementationCache* WebViewGetResourceLoadDelegateImplementations(WebView *);
WebFrameLoadDelegateImplementationCache* WebViewGetFrameLoadDelegateImplementations(WebView *);
WebScriptDebugDelegateImplementationCache* WebViewGetScriptDebugDelegateImplementations(WebView *);
WebHistoryDelegateImplementationCache* WebViewGetHistoryDelegateImplementations(WebView *webView);

id CallFormDelegate(WebView *, SEL, id, id);
id CallFormDelegate(WebView *, SEL, id, id, id);
id CallFormDelegate(WebView *self, SEL selector, id object1, id object2, id object3, id object4, id object5);
BOOL CallFormDelegateReturningBoolean(BOOL, WebView *, SEL, id, SEL, id);

id CallUIDelegate(WebView *, SEL);
id CallUIDelegate(WebView *, SEL, id);
id CallUIDelegate(WebView *, SEL, NSRect);
id CallUIDelegate(WebView *, SEL, id, id);
id CallUIDelegate(WebView *, SEL, id, BOOL);
id CallUIDelegate(WebView *, SEL, id, id, id);
id CallUIDelegate(WebView *, SEL, id, NSUInteger);
float CallUIDelegateReturningFloat(WebView *, SEL);
BOOL CallUIDelegateReturningBoolean(BOOL, WebView *, SEL);
BOOL CallUIDelegateReturningBoolean(BOOL, WebView *, SEL, id);
BOOL CallUIDelegateReturningBoolean(BOOL, WebView *, SEL, id, id);
BOOL CallUIDelegateReturningBoolean(BOOL, WebView *, SEL, id, BOOL);
BOOL CallUIDelegateReturningBoolean(BOOL, WebView *, SEL, id, BOOL, id);
#if PLATFORM(IOS_FAMILY)
BOOL CallUIDelegateReturningBoolean(BOOL, WebView *, SEL, id, id, BOOL);
#endif

id CallFrameLoadDelegate(IMP, WebView *, SEL);
id CallFrameLoadDelegate(IMP, WebView *, SEL, NSUInteger);
id CallFrameLoadDelegate(IMP, WebView *, SEL, id);
id CallFrameLoadDelegate(IMP, WebView *, SEL, id, id);
id CallFrameLoadDelegate(IMP, WebView *, SEL, id, id, id);
id CallFrameLoadDelegate(IMP, WebView *, SEL, id, id, id, id);
id CallFrameLoadDelegate(IMP, WebView *, SEL, id, NSTimeInterval, id, id);
#if PLATFORM(IOS_FAMILY)
id CallFrameLoadDelegate(IMP, WebView *, SEL, id, double);
id CallFrameLoadDelegateInWebThread(IMP, WebView *, SEL, NSUInteger);
#endif

BOOL CallFrameLoadDelegateReturningBoolean(BOOL, IMP, WebView *, SEL);

id CallResourceLoadDelegate(IMP, WebView *, SEL, id, id);
id CallResourceLoadDelegate(IMP, WebView *, SEL, id, id, id);
id CallResourceLoadDelegate(IMP, WebView *, SEL, id, id, id, id);
id CallResourceLoadDelegate(IMP, WebView *, SEL, id, NSInteger, id);
id CallResourceLoadDelegate(IMP, WebView *, SEL, id, id, NSInteger, id);
#if PLATFORM(IOS_FAMILY)
id CallResourceLoadDelegateInWebThread(IMP, WebView *, SEL, id, id);
id CallResourceLoadDelegateInWebThread(IMP, WebView *, SEL, id, id, id);
id CallResourceLoadDelegateInWebThread(IMP, WebView *, SEL, id, id, id, id);
id CallResourceLoadDelegateInWebThread(IMP, WebView *, SEL, id, NSInteger, id);
id CallResourceLoadDelegateInWebThread(IMP, WebView *, SEL, id, id, NSInteger, id);
#endif

BOOL CallResourceLoadDelegateReturningBoolean(BOOL, IMP, WebView *, SEL, id);
BOOL CallResourceLoadDelegateReturningBoolean(BOOL, IMP, WebView *, SEL, id, id);
BOOL CallResourceLoadDelegateReturningBoolean(BOOL, IMP, WebView *, SEL, id, id, id);

id CallScriptDebugDelegate(IMP, WebView *, SEL, id, id, NSInteger, id);
id CallScriptDebugDelegate(IMP, WebView *, SEL, id, NSInteger, id, NSInteger, id);
id CallScriptDebugDelegate(IMP, WebView *, SEL, id, NSInteger, id, id, id);
id CallScriptDebugDelegate(IMP, WebView *, SEL, id, NSInteger, int, id);
id CallScriptDebugDelegate(IMP, WebView *, SEL, id, BOOL, NSInteger, int, id);

id CallHistoryDelegate(IMP, WebView *, SEL);
id CallHistoryDelegate(IMP, WebView *, SEL, id, id);
id CallHistoryDelegate(IMP, WebView *, SEL, id, id, id);

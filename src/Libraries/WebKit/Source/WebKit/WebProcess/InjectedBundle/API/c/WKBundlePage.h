/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#ifndef WKBundlePage_h
#define WKBundlePage_h

#include <JavaScriptCore/JavaScript.h>
#include <WebKit/WKBase.h>
#include <WebKit/WKBundlePageContextMenuClient.h>
#include <WebKit/WKBundlePageEditorClient.h>
#include <WebKit/WKBundlePageFormClient.h>
#include <WebKit/WKBundlePageLoaderClient.h>
#include <WebKit/WKBundlePagePolicyClient.h>
#include <WebKit/WKBundlePageResourceLoadClient.h>
#include <WebKit/WKBundlePageUIClient.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKFindOptions.h>
#include <WebKit/WKImage.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKBundlePageGetTypeID();

WK_EXPORT void WKBundlePageSetContextMenuClient(WKBundlePageRef page, WKBundlePageContextMenuClientBase* client);
WK_EXPORT void WKBundlePageSetEditorClient(WKBundlePageRef page, WKBundlePageEditorClientBase* client);
WK_EXPORT void WKBundlePageSetFormClient(WKBundlePageRef page, WKBundlePageFormClientBase* client);
WK_EXPORT void WKBundlePageSetPageLoaderClient(WKBundlePageRef page, WKBundlePageLoaderClientBase* client);
WK_EXPORT void WKBundlePageSetResourceLoadClient(WKBundlePageRef page, WKBundlePageResourceLoadClientBase* client);
WK_EXPORT void WKBundlePageSetPolicyClient(WKBundlePageRef page, WKBundlePagePolicyClientBase* client) WK_C_API_DEPRECATED;
WK_EXPORT void WKBundlePageSetUIClient(WKBundlePageRef page, WKBundlePageUIClientBase* client);

WK_EXPORT WKBundleFrameRef WKBundlePageGetMainFrame(WKBundlePageRef page) WK_C_API_DEPRECATED_WITH_MESSAGE("With site isolation, the main frame is not necessarily local to the current bundle process.");
WK_EXPORT WKFrameHandleRef WKBundleFrameCreateFrameHandle(WKBundleFrameRef);

WK_EXPORT WKBundleBackForwardListRef WKBundlePageGetBackForwardList(WKBundlePageRef page) WK_C_API_DEPRECATED;
WK_EXPORT WKStringRef WKBundlePageDumpHistoryForTesting(WKBundlePageRef page, WKStringRef directory);

WK_EXPORT void WKBundlePageInstallPageOverlay(WKBundlePageRef page, WKBundlePageOverlayRef pageOverlay);
WK_EXPORT void WKBundlePageUninstallPageOverlay(WKBundlePageRef page, WKBundlePageOverlayRef pageOverlay);

WK_EXPORT void WKBundlePageInstallPageOverlayWithAnimation(WKBundlePageRef page, WKBundlePageOverlayRef pageOverlay);
WK_EXPORT void WKBundlePageUninstallPageOverlayWithAnimation(WKBundlePageRef page, WKBundlePageOverlayRef pageOverlay);

WK_EXPORT void WKBundlePageSetTopOverhangImage(WKBundlePageRef page, WKImageRef image);
WK_EXPORT void WKBundlePageSetBottomOverhangImage(WKBundlePageRef page, WKImageRef image);

#if !defined(TARGET_OS_IPHONE) || !TARGET_OS_IPHONE
WK_EXPORT void WKBundlePageSetHeaderBanner(WKBundlePageRef page, WKBundlePageBannerRef banner);
WK_EXPORT void WKBundlePageSetFooterBanner(WKBundlePageRef page, WKBundlePageBannerRef banner);
#endif // !TARGET_OS_IPHONE

WK_EXPORT bool WKBundlePageHasLocalDataForURL(WKBundlePageRef page, WKURLRef url);
WK_EXPORT bool WKBundlePageCanHandleRequest(WKURLRequestRef request);

WK_EXPORT bool WKBundlePageFindString(WKBundlePageRef page, WKStringRef target, WKFindOptions findOptions);
WK_EXPORT void WKBundlePageReplaceStringMatches(WKBundlePageRef page, WKArrayRef matchIndices, WKStringRef replacementText, bool selectionOnly);

WK_EXPORT WKImageRef WKBundlePageCreateSnapshotWithOptions(WKBundlePageRef page, WKRect rect, WKSnapshotOptions options);

// We should deprecate these functions in favor of just using WKBundlePageCreateSnapshotWithOptions.
WK_EXPORT WKImageRef WKBundlePageCreateSnapshotInViewCoordinates(WKBundlePageRef page, WKRect rect, WKImageOptions options);
WK_EXPORT WKImageRef WKBundlePageCreateSnapshotInDocumentCoordinates(WKBundlePageRef page, WKRect rect, WKImageOptions options);

// We should keep this function since it allows passing a scale factor, but we should re-name it to
// WKBundlePageCreateScaledSnapshotWithOptions.
WK_EXPORT WKImageRef WKBundlePageCreateScaledSnapshotInDocumentCoordinates(WKBundlePageRef page, WKRect rect, double scaleFactor, WKImageOptions options);

WK_EXPORT double WKBundlePageGetBackingScaleFactor(WKBundlePageRef page);

WK_EXPORT void WKBundlePageListenForLayoutMilestones(WKBundlePageRef page, WKLayoutMilestones milestones);

WK_EXPORT void WKBundlePageCloseInspectorForTest(WKBundlePageRef page);
WK_EXPORT void WKBundlePageEvaluateScriptInInspectorForTest(WKBundlePageRef page, WKStringRef script);

WK_EXPORT bool WKBundlePageIsUsingEphemeralSession(WKBundlePageRef page);

WK_EXPORT bool WKBundlePageIsControlledByAutomation(WKBundlePageRef page);

WK_EXPORT void WKBundlePageStartMonitoringScrollOperations(WKBundlePageRef page, bool clearLatchingState);

WK_EXPORT WKStringRef WKBundlePageCopyGroupIdentifier(WKBundlePageRef page);

WK_EXPORT void WKBundlePageSetSkipDecidePolicyForResponseIfPossible(WKBundlePageRef page, bool skip);

WK_EXPORT WKStringRef WKBundlePageCopyFrameTextForTesting(WKBundlePageRef page, bool includeSubframes);

typedef void (*WKBundlePageTestNotificationCallback)(void* context);
// Returns true  if the callback function will be called, else false.
WK_EXPORT bool WKBundlePageRegisterScrollOperationCompletionCallback(WKBundlePageRef, WKBundlePageTestNotificationCallback, bool expectWheelEndOrCancel, bool expectMomentumEnd, void* context);

// Call the given callback after both a postTask() on the page's document's ScriptExecutionContext, and a zero-delay timer.
WK_EXPORT void WKBundlePageCallAfterTasksAndTimers(WKBundlePageRef, WKBundlePageTestNotificationCallback, void* context);

WK_EXPORT void WKBundlePageLayoutIfNeeded(WKBundlePageRef page);

WK_EXPORT void WKBundlePagePostMessage(WKBundlePageRef page, WKStringRef messageName, WKTypeRef messageBody);

typedef void (*WKBundlePageMessageReplyCallback)(WKTypeRef reply, void* context);
WK_EXPORT void WKBundlePagePostMessageWithAsyncReply(WKBundlePageRef page, WKStringRef messageName, WKTypeRef messageBody, WKBundlePageMessageReplyCallback replyCallback, void* context);

// Switches a connection into a fully synchronous mode, so all messages become synchronous until we get a response.
WK_EXPORT void WKBundlePagePostSynchronousMessageForTesting(WKBundlePageRef page, WKStringRef messageName, WKTypeRef messageBody, WKTypeRef* returnRetainedData);
// Same as WKBundlePagePostMessage() but the message cannot become synchronous, even if the connection is in fully synchronous mode.
WK_EXPORT void WKBundlePagePostMessageIgnoringFullySynchronousMode(WKBundlePageRef page, WKStringRef messageName, WKTypeRef messageBody);

WK_EXPORT void WKBundlePageFlushDeferredDidReceiveMouseEventForTesting(WKBundlePageRef page);

#ifdef __cplusplus
}
#endif

#endif /* WKBundlePage_h */

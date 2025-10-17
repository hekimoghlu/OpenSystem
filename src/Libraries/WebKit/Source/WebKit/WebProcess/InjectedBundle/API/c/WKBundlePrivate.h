/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
#ifndef WKBundlePrivate_h
#define WKBundlePrivate_h

#include <WebKit/WKBase.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <JavaScriptCore/JSBase.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKUserContentInjectedFrames.h>
#include <WebKit/WKUserScriptInjectionTime.h>

#ifdef __cplusplus
extern "C" {
#endif

// TestRunner only SPIs.
WK_EXPORT void WKBundleAddOriginAccessAllowListEntry(WKBundleRef bundle, WKStringRef, WKStringRef, WKStringRef, bool);
WK_EXPORT void WKBundleRemoveOriginAccessAllowListEntry(WKBundleRef bundle, WKStringRef, WKStringRef, WKStringRef, bool);
WK_EXPORT void WKBundleResetOriginAccessAllowLists(WKBundleRef bundle);
WK_EXPORT int WKBundleNumberOfPages(WKBundleRef bundle, WKBundleFrameRef frameRef, double pageWidthInPixels, double pageHeightInPixels);
WK_EXPORT int WKBundlePageNumberForElementById(WKBundleRef bundle, WKBundleFrameRef frameRef, WKStringRef idRef, double pageWidthInPixels, double pageHeightInPixels);
WK_EXPORT WKStringRef WKBundlePageSizeAndMarginsInPixels(WKBundleRef bundle, WKBundleFrameRef frameRef, int, int, int, int, int, int, int);
WK_EXPORT bool WKBundleIsPageBoxVisible(WKBundleRef bundle, WKBundleFrameRef frameRef, int);
WK_EXPORT void WKBundleSetUserStyleSheetLocationForTesting(WKBundleRef bundle, WKStringRef location);
WK_EXPORT void WKBundleRemoveAllWebNotificationPermissions(WKBundleRef bundle, WKBundlePageRef page);
WK_EXPORT WKDataRef WKBundleCopyWebNotificationID(WKBundleRef bundle, JSContextRef context, JSValueRef notification);
WK_EXPORT WKDataRef WKBundleCreateWKDataFromUInt8Array(WKBundleRef bundle, JSContextRef context, JSValueRef data);
WK_EXPORT void WKBundleSetAsynchronousSpellCheckingEnabledForTesting(WKBundleRef bundleRef, bool enabled);
// Returns array of dictionaries. Dictionary keys are document identifiers, values are document URLs.
WK_EXPORT WKArrayRef WKBundleGetLiveDocumentURLsForTesting(WKBundleRef bundle, bool excludeDocumentsInPageGroupPages);

// Local storage API
WK_EXPORT void WKBundleSetDatabaseQuota(WKBundleRef bundle, uint64_t);

// Garbage collection API
WK_EXPORT void WKBundleGarbageCollectJavaScriptObjects(WKBundleRef bundle);
WK_EXPORT void WKBundleGarbageCollectJavaScriptObjectsOnAlternateThreadForDebugging(WKBundleRef bundle, bool waitUntilDone);
WK_EXPORT size_t WKBundleGetJavaScriptObjectsCount(WKBundleRef bundle);

WK_EXPORT bool WKBundleIsProcessingUserGesture(WKBundleRef bundle);

WK_EXPORT void WKBundleSetTabKeyCyclesThroughElements(WKBundleRef bundle, WKBundlePageRef page, bool enabled);

WK_EXPORT void WKBundleClearResourceLoadStatistics(WKBundleRef bundle);

typedef void (*NotifyObserverCallback)(void* functionContext);
WK_EXPORT void WKBundleResourceLoadStatisticsNotifyObserver(WKBundleRef bundle, void* context, NotifyObserverCallback callback);

WK_EXPORT void WKBundleExtendClassesForParameterCoder(WKBundleRef bundle, WKArrayRef classes);

WK_EXPORT void WKBundleReleaseMemory(WKBundleRef bundle);

#ifdef __cplusplus
}
#endif

#endif /* WKBundlePrivate_h */

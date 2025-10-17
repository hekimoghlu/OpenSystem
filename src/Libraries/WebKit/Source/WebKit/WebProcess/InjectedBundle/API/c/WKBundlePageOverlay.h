/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
#ifndef WKBundlePageOverlay_h
#define WKBundlePageOverlay_h

#include <WebKit/WKBase.h>
#include <WebKit/WKEvent.h>
#include <WebKit/WKGeometry.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Page overlay client.
typedef void (*WKBundlePageOverlayWillMoveToPageCallback)(WKBundlePageOverlayRef pageOverlay, WKBundlePageRef page, const void* clientInfo);
typedef void (*WKBundlePageOverlayDidMoveToPageCallback)(WKBundlePageOverlayRef pageOverlay, WKBundlePageRef page, const void* clientInfo);
typedef void (*WKBundlePageOverlayDrawRectCallback)(WKBundlePageOverlayRef pageOverlay, void* graphicsContext, WKRect dirtyRect, const void* clientInfo);
typedef bool (*WKBundlePageOverlayMouseDownCallback)(WKBundlePageOverlayRef pageOverlay, WKPoint position, WKEventMouseButton mouseButton, const void* clientInfo);
typedef bool (*WKBundlePageOverlayMouseUpCallback)(WKBundlePageOverlayRef pageOverlay, WKPoint position, WKEventMouseButton mouseButton, const void* clientInfo);
typedef bool (*WKBundlePageOverlayMouseMovedCallback)(WKBundlePageOverlayRef pageOverlay, WKPoint position, const void* clientInfo);
typedef bool (*WKBundlePageOverlayMouseDraggedCallback)(WKBundlePageOverlayRef pageOverlay, WKPoint position, WKEventMouseButton mouseButton, const void* clientInfo);

typedef void* (*WKBundlePageOverlayActionContextForResultAtPointCallback)(WKBundlePageOverlayRef pageOverlay, WKPoint position, WKBundleRangeHandleRef* rangeHandle, const void* clientInfo);
typedef void (*WKBundlePageOverlayDataDetectorsDidPresentUI)(WKBundlePageOverlayRef pageOverlay, const void* clientInfo);
typedef void (*WKBundlePageOverlayDataDetectorsDidChangeUI)(WKBundlePageOverlayRef pageOverlay, const void* clientInfo);
typedef void (*WKBundlePageOverlayDataDetectorsDidHideUI)(WKBundlePageOverlayRef pageOverlay, const void* clientInfo);

typedef struct WKBundlePageOverlayClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageOverlayClientBase;

typedef struct WKBundlePageOverlayClientV0 {
    WKBundlePageOverlayClientBase                                       base;

    WKBundlePageOverlayWillMoveToPageCallback                           willMoveToPage;
    WKBundlePageOverlayDidMoveToPageCallback                            didMoveToPage;
    WKBundlePageOverlayDrawRectCallback                                 drawRect;
    WKBundlePageOverlayMouseDownCallback                                mouseDown;
    WKBundlePageOverlayMouseUpCallback                                  mouseUp;
    WKBundlePageOverlayMouseMovedCallback                               mouseMoved;
    WKBundlePageOverlayMouseDraggedCallback                             mouseDragged;
} WKBundlePageOverlayClientV0;

typedef struct WKBundlePageOverlayClientV1 {
    WKBundlePageOverlayClientBase                                       base;

    WKBundlePageOverlayWillMoveToPageCallback                           willMoveToPage;
    WKBundlePageOverlayDidMoveToPageCallback                            didMoveToPage;
    WKBundlePageOverlayDrawRectCallback                                 drawRect;
    WKBundlePageOverlayMouseDownCallback                                mouseDown;
    WKBundlePageOverlayMouseUpCallback                                  mouseUp;
    WKBundlePageOverlayMouseMovedCallback                               mouseMoved;
    WKBundlePageOverlayMouseDraggedCallback                             mouseDragged;

    WKBundlePageOverlayActionContextForResultAtPointCallback            actionContextForResultAtPoint;
    WKBundlePageOverlayDataDetectorsDidPresentUI                        dataDetectorsDidPresentUI;
    WKBundlePageOverlayDataDetectorsDidChangeUI                         dataDetectorsDidChangeUI;
    WKBundlePageOverlayDataDetectorsDidHideUI                           dataDetectorsDidHideUI;
} WKBundlePageOverlayClientV1;

typedef WKTypeRef (*WKAccessibilityAttributeValueCallback)(WKBundlePageOverlayRef pageOverlay, WKStringRef attribute, WKTypeRef parameter, const void* clientInfo);
typedef WKArrayRef (*WKAccessibilityAttributeNamesCallback)(WKBundlePageOverlayRef pageOverlay, bool parameterizedNames, const void* clientInfo);

typedef struct WKBundlePageOverlayAccessibilityClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageOverlayAccessibilityClientBase;

typedef struct WKBundlePageOverlayAccessibilityClientV0 {
    WKBundlePageOverlayAccessibilityClientBase                          base;

    // Version 0.
    WKAccessibilityAttributeValueCallback                               copyAccessibilityAttributeValue;
    WKAccessibilityAttributeNamesCallback                               copyAccessibilityAttributeNames;
} WKBundlePageOverlayAccessibilityClientV0;

WK_EXPORT WKTypeID WKBundlePageOverlayGetTypeID();

WK_EXPORT WKBundlePageOverlayRef WKBundlePageOverlayCreate(WKBundlePageOverlayClientBase* client);
WK_EXPORT void WKBundlePageOverlaySetNeedsDisplay(WKBundlePageOverlayRef bundlePageOverlay, WKRect rect);
WK_EXPORT float WKBundlePageOverlayFractionFadedIn(WKBundlePageOverlayRef bundlePageOverlay);
WK_EXPORT void WKBundlePageOverlaySetAccessibilityClient(WKBundlePageOverlayRef bundlePageOverlay, WKBundlePageOverlayAccessibilityClientBase* client);
WK_EXPORT void WKBundlePageOverlayClear(WKBundlePageOverlayRef bundlePageOverlay);

#ifdef __cplusplus
}
#endif

#endif // WKBundlePageOverlay_h

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#ifndef WKBundlePageBanner_h
#define WKBundlePageBanner_h

#include <WebKit/WKBase.h>
#include <WebKit/WKEvent.h>
#include <WebKit/WKGeometry.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Page banner client.
typedef bool (*WKBundlePageBannerMouseDownCallback)(WKBundlePageBannerRef pageBanner, WKPoint position, WKEventMouseButton mouseButton, const void* clientInfo);
typedef bool (*WKBundlePageBannerMouseUpCallback)(WKBundlePageBannerRef pageBanner, WKPoint position, WKEventMouseButton mouseButton, const void* clientInfo);
typedef bool (*WKBundlePageBannerMouseMovedCallback)(WKBundlePageBannerRef pageBanner, WKPoint position, const void* clientInfo);
typedef bool (*WKBundlePageBannerMouseDraggedCallback)(WKBundlePageBannerRef pageBanner, WKPoint position, WKEventMouseButton mouseButton, const void* clientInfo);

typedef struct WKBundlePageBannerClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageBannerClientBase;

typedef struct WKBundlePageBannerClientV0 {
    WKBundlePageBannerClientBase                                        base;

    // Version 0.
    WKBundlePageBannerMouseDownCallback                                 mouseDown;
    WKBundlePageBannerMouseUpCallback                                   mouseUp;
    WKBundlePageBannerMouseMovedCallback                                mouseMoved;
    WKBundlePageBannerMouseDraggedCallback                              mouseDragged;
} WKBundlePageBannerClientV0;

WK_EXPORT WKTypeID WKBundlePageBannerGetTypeID();

#ifdef __cplusplus
}
#endif

#endif // WKBundlePageBanner_h

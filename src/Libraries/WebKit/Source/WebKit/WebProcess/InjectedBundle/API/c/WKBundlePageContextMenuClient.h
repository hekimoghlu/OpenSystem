/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#ifndef WKBundlePageContextMenuClient_h
#define WKBundlePageContextMenuClient_h

#include <WebKit/WKBase.h>

typedef void (*WKBundlePageGetContextMenuFromDefaultContextMenuCallback)(WKBundlePageRef page, WKBundleHitTestResultRef hitTestResult, WKArrayRef defaultMenu, WKArrayRef* newMenu, WKTypeRef* userData, const void* clientInfo);
typedef void (*WKBundlePagePrepareForActionMenuCallback)(WKBundlePageRef page, WKBundleHitTestResultRef hitTestResult, WKTypeRef* userData, const void* clientInfo);

typedef struct WKBundlePageContextMenuClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageContextMenuClientBase;

typedef struct WKBundlePageContextMenuClientV0 {
    WKBundlePageContextMenuClientBase                                   base;

    WKBundlePageGetContextMenuFromDefaultContextMenuCallback            getContextMenuFromDefaultMenu;
} WKBundlePageContextMenuClientV0;

typedef struct WKBundlePageContextMenuClientV1 {
    WKBundlePageContextMenuClientBase                                   base;

    WKBundlePageGetContextMenuFromDefaultContextMenuCallback            getContextMenuFromDefaultMenu;

    // This is actually about immediate actions; we should consider deprecating and renaming.
    WKBundlePagePrepareForActionMenuCallback                            prepareForActionMenu;
} WKBundlePageContextMenuClientV1;

#endif // WKBundlePageContextMenuClient_h

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#ifndef WKIconDatabase_h
#define WKIconDatabase_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

// IconDatabase Client.
typedef void (*WKIconDatabaseDidChangeIconForPageURLCallback)(WKIconDatabaseRef iconDatabase, WKURLRef pageURL, const void* clientInfo);
typedef void (*WKIconDatabaseDidRemoveAllIconsCallback)(WKIconDatabaseRef iconDatabase, const void* clientInfo);
typedef void (*WKIconDatabaseIconDataReadyForPageURLCallback)(WKIconDatabaseRef iconDatabase, WKURLRef pageURL, const void* clientInfo);

typedef struct WKIconDatabaseClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKIconDatabaseClientBase;

typedef struct WKIconDatabaseClientV0 {
    WKIconDatabaseClientBase                                            base;

    // Version 0.
    WKIconDatabaseDidChangeIconForPageURLCallback                       didChangeIconForPageURL;
    WKIconDatabaseDidRemoveAllIconsCallback                             didRemoveAllIcons;
} WKIconDatabaseClientV0;

typedef struct WKIconDatabaseClientV1 {
    WKIconDatabaseClientBase                                            base;

    // Version 0.
    WKIconDatabaseDidChangeIconForPageURLCallback                       didChangeIconForPageURL;
    WKIconDatabaseDidRemoveAllIconsCallback                             didRemoveAllIcons;

    // Version 1.
    WKIconDatabaseIconDataReadyForPageURLCallback                       iconDataReadyForPageURL;
} WKIconDatabaseClientV1;

WK_EXPORT WKTypeID WKIconDatabaseGetTypeID();

WK_EXPORT void WKIconDatabaseSetIconDatabaseClient(WKIconDatabaseRef iconDatabase, const WKIconDatabaseClientBase* client);

WK_EXPORT void WKIconDatabaseRetainIconForURL(WKIconDatabaseRef iconDatabase, WKURLRef pageURL);
WK_EXPORT void WKIconDatabaseReleaseIconForURL(WKIconDatabaseRef iconDatabase, WKURLRef pageURL);
WK_EXPORT void WKIconDatabaseSetIconDataForIconURL(WKIconDatabaseRef iconDatabase, WKDataRef iconData, WKURLRef iconURL);
WK_EXPORT void WKIconDatabaseSetIconURLForPageURL(WKIconDatabaseRef iconDatabase, WKURLRef iconURL, WKURLRef pageURL);
WK_EXPORT WKURLRef WKIconDatabaseCopyIconURLForPageURL(WKIconDatabaseRef iconDatabase, WKURLRef pageURL);
WK_EXPORT WKDataRef WKIconDatabaseCopyIconDataForPageURL(WKIconDatabaseRef iconDatabase, WKURLRef pageURL);

WK_EXPORT void WKIconDatabaseEnableDatabaseCleanup(WKIconDatabaseRef iconDatabase);

WK_EXPORT void WKIconDatabaseRemoveAllIcons(WKIconDatabaseRef iconDatabase);
WK_EXPORT void WKIconDatabaseCheckIntegrityBeforeOpening(WKIconDatabaseRef iconDatabase);

WK_EXPORT void WKIconDatabaseClose(WKIconDatabaseRef iconDatabase);

#ifdef __cplusplus
}
#endif

#endif /* WKIconDatabase_h */

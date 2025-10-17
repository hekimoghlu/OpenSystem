/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef WKPageContextMenuClient_h
#define WKPageContextMenuClient_h

#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKPageGetContextMenuFromProposedContextMenuCallback)(WKPageRef page, WKArrayRef proposedMenu, WKArrayRef* newMenu, WKHitTestResultRef hitTestResult, WKTypeRef userData, const void* clientInfo);
typedef void (*WKPageGetContextMenuFromProposedContextMenuCallbackAsync)(WKPageRef page, WKArrayRef proposedMenu, WKContextMenuListenerRef listener, WKHitTestResultRef hitTestResult, WKTypeRef userData, const void* clientInfo);
typedef void (*WKPageCustomContextMenuItemSelectedCallback)(WKPageRef page, WKContextMenuItemRef contextMenuItem, const void* clientInfo);
typedef void (*WKPageContextMenuDismissedCallback)(WKPageRef page, const void* clientInfo);
typedef void (*WKPageShowContextMenuCallback)(WKPageRef page, WKPoint menuLocation, WKArrayRef menuItems, const void* clientInfo);
typedef void (*WKPageHideContextMenuCallback)(WKPageRef page, const void* clientInfo);

// Deprecated
typedef void (*WKPageGetContextMenuFromProposedContextMenuCallback_deprecatedForUseWithV0)(WKPageRef page, WKArrayRef proposedMenu, WKArrayRef* newMenu, WKTypeRef userData, const void* clientInfo);

typedef struct WKPageContextMenuClientBase {
    int                                                                          version;
    const void *                                                                 clientInfo;
} WKPageContextMenuClientBase;

typedef struct WKPageContextMenuClientV0 {
    WKPageContextMenuClientBase                                                  base;

    // Version 0.
    WKPageGetContextMenuFromProposedContextMenuCallback_deprecatedForUseWithV0   getContextMenuFromProposedMenu_deprecatedForUseWithV0;
    WKPageCustomContextMenuItemSelectedCallback                                  customContextMenuItemSelected;
} WKPageContextMenuClientV0;

typedef struct WKPageContextMenuClientV1 {
    WKPageContextMenuClientBase                                                  base;

    // Version 0.
    WKPageGetContextMenuFromProposedContextMenuCallback_deprecatedForUseWithV0   getContextMenuFromProposedMenu_deprecatedForUseWithV0;
    WKPageCustomContextMenuItemSelectedCallback                                  customContextMenuItemSelected;

    // Version 1.
    WKPageContextMenuDismissedCallback                                           contextMenuDismissed;
} WKPageContextMenuClientV1;

typedef struct WKPageContextMenuClientV2 {
    WKPageContextMenuClientBase                                                  base;

    // Version 0.
    WKPageGetContextMenuFromProposedContextMenuCallback_deprecatedForUseWithV0   getContextMenuFromProposedMenu_deprecatedForUseWithV0;
    WKPageCustomContextMenuItemSelectedCallback                                  customContextMenuItemSelected;

    // Version 1.
    WKPageContextMenuDismissedCallback                                           contextMenuDismissed;

    // Version 2.
    WKPageGetContextMenuFromProposedContextMenuCallback                          getContextMenuFromProposedMenu;
} WKPageContextMenuClientV2;

typedef struct WKPageContextMenuClientV3 {
    WKPageContextMenuClientBase                                                  base;

    // Version 0.
    WKPageGetContextMenuFromProposedContextMenuCallback_deprecatedForUseWithV0   getContextMenuFromProposedMenu_deprecatedForUseWithV0;
    WKPageCustomContextMenuItemSelectedCallback                                  customContextMenuItemSelected;

    // Version 1.
    WKPageContextMenuDismissedCallback                                           contextMenuDismissed;

    // Version 2.
    WKPageGetContextMenuFromProposedContextMenuCallback                          getContextMenuFromProposedMenu;

    // Version 3.
    WKPageShowContextMenuCallback                                                showContextMenu;
    WKPageHideContextMenuCallback                                                hideContextMenu;
} WKPageContextMenuClientV3;

typedef struct WKPageContextMenuClientV4 {
    WKPageContextMenuClientBase                                                  base;

    // Version 0.
    WKPageGetContextMenuFromProposedContextMenuCallback_deprecatedForUseWithV0   getContextMenuFromProposedMenu_deprecatedForUseWithV0;
    WKPageCustomContextMenuItemSelectedCallback                                  customContextMenuItemSelected;

    // Version 1.
    WKPageContextMenuDismissedCallback                                           contextMenuDismissed;

    // Version 2.
    WKPageGetContextMenuFromProposedContextMenuCallback                          getContextMenuFromProposedMenu;

    // Version 3.
    WKPageShowContextMenuCallback                                                showContextMenu;
    WKPageHideContextMenuCallback                                                hideContextMenu;

    // Version 4.
    WKPageGetContextMenuFromProposedContextMenuCallbackAsync                     getContextMenuFromProposedMenuAsync;

} WKPageContextMenuClientV4;

#ifdef __cplusplus
}
#endif


#endif // WKPageContextMenuClient_h

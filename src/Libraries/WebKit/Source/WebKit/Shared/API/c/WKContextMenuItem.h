/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#ifndef WKContextMenuItem_h
#define WKContextMenuItem_h

#include <WebKit/WKBase.h>
#include <WebKit/WKContextMenuItemTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKContextMenuItemGetTypeID(void);

WK_EXPORT WKContextMenuItemRef WKContextMenuItemCreateAsAction(WKContextMenuItemTag, WKStringRef title, bool enabled);
WK_EXPORT WKContextMenuItemRef WKContextMenuItemCreateAsCheckableAction(WKContextMenuItemTag, WKStringRef title, bool enabled, bool checked);
WK_EXPORT WKContextMenuItemRef WKContextMenuItemCreateAsSubmenu(WKStringRef title, bool enabled, WKArrayRef submenuItems);
WK_EXPORT WKContextMenuItemRef WKContextMenuItemSeparatorItem();

WK_EXPORT WKContextMenuItemTag WKContextMenuItemGetTag(WKContextMenuItemRef);
WK_EXPORT WKContextMenuItemType WKContextMenuItemGetType(WKContextMenuItemRef);
WK_EXPORT WKStringRef WKContextMenuItemCopyTitle(WKContextMenuItemRef);
WK_EXPORT bool WKContextMenuItemGetEnabled(WKContextMenuItemRef);
WK_EXPORT bool WKContextMenuItemGetChecked(WKContextMenuItemRef);
WK_EXPORT WKArrayRef WKContextMenuCopySubmenuItems(WKContextMenuItemRef);

WK_EXPORT WKTypeRef WKContextMenuItemGetUserData(WKContextMenuItemRef);
WK_EXPORT void WKContextMenuItemSetUserData(WKContextMenuItemRef, WKTypeRef);

#ifdef __cplusplus
}
#endif

#endif /* WKContextMenuItem_h */

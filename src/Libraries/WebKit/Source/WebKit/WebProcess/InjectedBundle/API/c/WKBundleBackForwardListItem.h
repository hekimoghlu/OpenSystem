/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#ifndef WKBundleBackForwardListItem_h
#define WKBundleBackForwardListItem_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKBundleBackForwardListItemGetTypeID() WK_C_API_DEPRECATED;

WK_EXPORT bool WKBundleBackForwardListItemIsSame(WKBundleBackForwardListItemRef item1, WKBundleBackForwardListItemRef item2) WK_C_API_DEPRECATED;

WK_EXPORT WKURLRef WKBundleBackForwardListItemCopyOriginalURL(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;
WK_EXPORT WKURLRef WKBundleBackForwardListItemCopyURL(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;
WK_EXPORT WKStringRef WKBundleBackForwardListItemCopyTitle(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;

WK_EXPORT WKStringRef WKBundleBackForwardListItemCopyTarget(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;
WK_EXPORT bool WKBundleBackForwardListItemIsTargetItem(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;

WK_EXPORT bool WKBundleBackForwardListItemIsInBackForwardCache(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;
WK_EXPORT bool WKBundleBackForwardListItemHasCachedPageExpired(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;

WK_EXPORT WKArrayRef WKBundleBackForwardListItemCopyChildren(WKBundleBackForwardListItemRef item) WK_C_API_DEPRECATED;

#ifdef __cplusplus
}
#endif

#endif /* WKBundleBackForwardListItem_h */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#ifndef WKBackForwardListRef_h
#define WKBackForwardListRef_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKBackForwardListGetTypeID(void);

WK_EXPORT WKBackForwardListItemRef WKBackForwardListGetCurrentItem(WKBackForwardListRef list);
WK_EXPORT WKBackForwardListItemRef WKBackForwardListGetBackItem(WKBackForwardListRef list);
WK_EXPORT WKBackForwardListItemRef WKBackForwardListGetForwardItem(WKBackForwardListRef list);
WK_EXPORT WKBackForwardListItemRef WKBackForwardListGetItemAtIndex(WKBackForwardListRef list, int index);

WK_EXPORT void WKBackForwardListClear(WKBackForwardListRef list);

WK_EXPORT unsigned WKBackForwardListGetBackListCount(WKBackForwardListRef list);
WK_EXPORT unsigned WKBackForwardListGetForwardListCount(WKBackForwardListRef list);

WK_EXPORT WKArrayRef WKBackForwardListCopyBackListWithLimit(WKBackForwardListRef list, unsigned limit);
WK_EXPORT WKArrayRef WKBackForwardListCopyForwardListWithLimit(WKBackForwardListRef list, unsigned limit);

#ifdef __cplusplus
}
#endif

#endif // WKBackForwardListRef_h

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#ifndef WKNavigationActionRef_h
#define WKNavigationActionRef_h

#include <WebKit/WKBase.h>
#include <WebKit/WKPageLoadTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKNavigationActionGetTypeID();

WK_EXPORT bool WKNavigationActionShouldPerformDownload(WKNavigationActionRef action);
WK_EXPORT WKURLRequestRef WKNavigationActionCopyRequest(WKNavigationActionRef action);
WK_EXPORT bool WKNavigationActionGetShouldOpenExternalSchemes(WKNavigationActionRef action);
WK_EXPORT WKFrameInfoRef WKNavigationActionCopyTargetFrameInfo(WKNavigationActionRef action);
WK_EXPORT WKFrameNavigationType WKNavigationActionGetNavigationType(WKNavigationActionRef action);
WK_EXPORT bool WKNavigationActionHasUnconsumedUserGesture(WKNavigationActionRef action);

#ifdef __cplusplus
}
#endif

#endif // WKNavigationActionRef_h

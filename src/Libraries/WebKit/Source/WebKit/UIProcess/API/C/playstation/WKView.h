/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#pragma once

#include <WebKit/WKViewClient.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wpe_view_backend;

WK_EXPORT WKViewRef WKViewCreate(WKPageConfigurationRef configuration);
WK_EXPORT WKViewRef WKViewCreateWPE(struct wpe_view_backend*, WKPageConfigurationRef);

WK_EXPORT void WKViewSetViewClient(WKViewRef, const WKViewClientBase*);

WK_EXPORT WKPageRef WKViewGetPage(WKViewRef);

WK_EXPORT void WKViewSetSize(WKViewRef, WKSize viewSize);

WK_EXPORT void WKViewSetFocus(WKViewRef, bool);
WK_EXPORT void WKViewSetActive(WKViewRef, bool);
WK_EXPORT void WKViewSetVisible(WKViewRef, bool);

WK_EXPORT void WKViewWillEnterFullScreen(WKViewRef);
WK_EXPORT void WKViewDidEnterFullScreen(WKViewRef);
WK_EXPORT void WKViewWillExitFullScreen(WKViewRef);
WK_EXPORT void WKViewDidExitFullScreen(WKViewRef);
WK_EXPORT void WKViewRequestExitFullScreen(WKViewRef);
WK_EXPORT bool WKViewIsFullScreen(WKViewRef);

#ifdef __cplusplus
}
#endif

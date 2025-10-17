/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#ifndef WKOpenPanelResultListener_h
#define WKOpenPanelResultListener_h

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKOpenPanelResultListenerGetTypeID(void);

#if defined(__APPLE__) && TARGET_OS_IPHONE
WK_EXPORT void WKOpenPanelResultListenerChooseMediaFiles(WKOpenPanelResultListenerRef listenerRef, WKArrayRef fileURLsRef, WKStringRef displayString, WKDataRef iconImageDataRef);
#endif
WK_EXPORT void WKOpenPanelResultListenerChooseFiles(WKOpenPanelResultListenerRef listener, WKArrayRef fileURLs, WKArrayRef allowedMimeTypesRef);
WK_EXPORT void WKOpenPanelResultListenerCancel(WKOpenPanelResultListenerRef listener);

#ifdef __cplusplus
}
#endif

#endif /* WKOpenPanelResultListener_h */

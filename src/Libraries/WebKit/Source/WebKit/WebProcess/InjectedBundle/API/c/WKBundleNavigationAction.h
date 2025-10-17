/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#ifndef WKBundleNavigationAction_h
#define WKBundleNavigationAction_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKEvent.h>
#include <WebKit/WKPageLoadTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKBundleNavigationActionGetTypeID() WK_C_API_DEPRECATED;

WK_EXPORT WKFrameNavigationType WKBundleNavigationActionGetNavigationType(WKBundleNavigationActionRef navigationAction) WK_C_API_DEPRECATED;
WK_EXPORT WKEventModifiers WKBundleNavigationActionGetEventModifiers(WKBundleNavigationActionRef navigationAction) WK_C_API_DEPRECATED;
WK_EXPORT WKEventMouseButton WKBundleNavigationActionGetEventMouseButton(WKBundleNavigationActionRef navigationAction) WK_C_API_DEPRECATED;
WK_EXPORT WKBundleHitTestResultRef WKBundleNavigationActionCopyHitTestResult(WKBundleNavigationActionRef navigationAction) WK_C_API_DEPRECATED;
WK_EXPORT WKBundleNodeHandleRef WKBundleNavigationActionCopyFormElement(WKBundleNavigationActionRef navigationAction) WK_C_API_DEPRECATED;

#ifdef __cplusplus
}
#endif

#endif /* WKBundleNavigationAction_h */

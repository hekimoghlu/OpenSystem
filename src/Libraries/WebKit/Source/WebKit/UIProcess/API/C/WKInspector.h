/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#ifndef WKInspector_h
#define WKInspector_h

#include <WebKit/WKBase.h>

#if !TARGET_OS_IPHONE

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKInspectorGetTypeID();

WK_EXPORT WKPageRef WKInspectorGetPage(WKInspectorRef inspector);

WK_EXPORT bool WKInspectorIsConnected(WKInspectorRef inspector);
WK_EXPORT bool WKInspectorIsVisible(WKInspectorRef inspector);
WK_EXPORT bool WKInspectorIsFront(WKInspectorRef inspector);

WK_EXPORT void WKInspectorConnect(WKInspectorRef inspector);

WK_EXPORT void WKInspectorShow(WKInspectorRef inspector);
WK_EXPORT void WKInspectorHide(WKInspectorRef inspector);
WK_EXPORT void WKInspectorClose(WKInspectorRef inspector);

WK_EXPORT void WKInspectorShowConsole(WKInspectorRef inspector);
WK_EXPORT void WKInspectorShowResources(WKInspectorRef inspector);
WK_EXPORT void WKInspectorShowMainResourceForFrame(WKInspectorRef inspector, WKFrameRef frame);

WK_EXPORT bool WKInspectorIsAttached(WKInspectorRef inspector);
WK_EXPORT void WKInspectorAttach(WKInspectorRef inspector);
WK_EXPORT void WKInspectorDetach(WKInspectorRef inspector);

WK_EXPORT bool WKInspectorIsProfilingPage(WKInspectorRef inspector);
WK_EXPORT void WKInspectorTogglePageProfiling(WKInspectorRef inspector);

WK_EXPORT bool WKInspectorIsElementSelectionActive(WKInspectorRef inspector);
WK_EXPORT void WKInspectorToggleElementSelection(WKInspectorRef inspector);

#ifdef __cplusplus
}
#endif

#endif // !TARGET_OS_IPHONE

#endif // WKInspector_h

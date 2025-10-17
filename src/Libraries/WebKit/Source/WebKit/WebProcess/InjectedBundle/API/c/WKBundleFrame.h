/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef WKBundleFrame_h
#define WKBundleFrame_h

#include <JavaScriptCore/JavaScript.h>
#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKFrame.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKBundleFrameGetTypeID();

WK_EXPORT bool WKBundleFrameIsMainFrame(WKBundleFrameRef frame);
WK_EXPORT WKBundleFrameRef WKBundleFrameGetParentFrame(WKBundleFrameRef frame) WK_C_API_DEPRECATED_WITH_MESSAGE("Parent frame might not be in the same process");
WK_EXPORT WKArrayRef WKBundleFrameCopyChildFrames(WKBundleFrameRef frame) WK_C_API_DEPRECATED_WITH_MESSAGE("Child frames might not be in the same process");

WK_EXPORT WKStringRef WKBundleFrameCopyName(WKBundleFrameRef frame);
WK_EXPORT WKURLRef WKBundleFrameCopyURL(WKBundleFrameRef frame);
WK_EXPORT WKURLRef WKBundleFrameCopyProvisionalURL(WKBundleFrameRef frame);
WK_EXPORT WKSecurityOriginRef WKBundleFrameCopySecurityOrigin(WKBundleFrameRef frame);

WK_EXPORT WKFrameLoadState WKBundleFrameGetFrameLoadState(WKBundleFrameRef frame); 

WK_EXPORT JSGlobalContextRef WKBundleFrameGetJavaScriptContext(WKBundleFrameRef frame);
WK_EXPORT JSGlobalContextRef WKBundleFrameGetJavaScriptContextForWorld(WKBundleFrameRef frame, WKBundleScriptWorldRef world);

WK_EXPORT WKBundleFrameRef WKBundleFrameForJavaScriptContext(JSContextRef context);

WK_EXPORT JSValueRef WKBundleFrameGetJavaScriptWrapperForNodeForWorld(WKBundleFrameRef frame, WKBundleNodeHandleRef nodeHandle, WKBundleScriptWorldRef world);
WK_EXPORT JSValueRef WKBundleFrameGetJavaScriptWrapperForRangeForWorld(WKBundleFrameRef frame, WKBundleRangeHandleRef rangeHandle, WKBundleScriptWorldRef world);
WK_EXPORT JSValueRef WKBundleFrameGetJavaScriptWrapperForFileForWorld(WKBundleFrameRef frame, WKBundleFileHandleRef fileHandle, WKBundleScriptWorldRef world) WK_C_API_DEPRECATED;

WK_EXPORT WKBundlePageRef WKBundleFrameGetPage(WKBundleFrameRef frame);

WK_EXPORT bool WKBundleFrameAllowsFollowingLink(WKBundleFrameRef frame, WKURLRef url);

WK_EXPORT WKRect WKBundleFrameGetContentBounds(WKBundleFrameRef frame);
WK_EXPORT WKRect WKBundleFrameGetVisibleContentBounds(WKBundleFrameRef frame);
WK_EXPORT WKRect WKBundleFrameGetVisibleContentBoundsExcludingScrollbars(WKBundleFrameRef frame);
WK_EXPORT WKSize WKBundleFrameGetScrollOffset(WKBundleFrameRef frame);

WK_EXPORT bool WKBundleFrameHasHorizontalScrollbar(WKBundleFrameRef frame);
WK_EXPORT bool WKBundleFrameHasVerticalScrollbar(WKBundleFrameRef frame);

WK_EXPORT bool WKBundleFrameGetDocumentBackgroundColor(WKBundleFrameRef frame, double* red, double* green, double* blue, double* alpha);

WK_EXPORT WKStringRef WKBundleFrameCopySuggestedFilenameForResourceWithURL(WKBundleFrameRef frame, WKURLRef url);
WK_EXPORT WKStringRef WKBundleFrameCopyMIMETypeForResourceWithURL(WKBundleFrameRef frame, WKURLRef url);

WK_EXPORT void WKBundleFrameSetAccessibleName(WKBundleFrameRef frame, WKStringRef accessibleName);

WK_EXPORT WKDataRef WKBundleFrameCopyWebArchive(WKBundleFrameRef frame);

typedef bool (*WKBundleFrameFrameFilterCallback)(WKBundleFrameRef frame, WKBundleFrameRef subframe, void* context);
WK_EXPORT WKDataRef WKBundleFrameCopyWebArchiveFilteringSubframes(WKBundleFrameRef frame, WKBundleFrameFrameFilterCallback frameFilterCallback, void* context);

#ifdef __cplusplus
}
#endif

#endif /* WKBundleFrame_h */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#ifndef WKFrame_h
#define WKFrame_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKFrameLoadStateProvisional = 0,
    kWKFrameLoadStateCommitted = 1,
    kWKFrameLoadStateFinished = 2
};
typedef uint32_t WKFrameLoadState;

WK_EXPORT WKTypeID WKFrameGetTypeID(void);
 
WK_EXPORT bool WKFrameIsMainFrame(WKFrameRef frame);
WK_EXPORT WKFrameLoadState WKFrameGetFrameLoadState(WKFrameRef frame);
WK_EXPORT WKURLRef WKFrameCopyProvisionalURL(WKFrameRef frame);
WK_EXPORT WKURLRef WKFrameCopyURL(WKFrameRef frame);
WK_EXPORT WKURLRef WKFrameCopyUnreachableURL(WKFrameRef frame);

WK_EXPORT void WKFrameStopLoading(WKFrameRef frame) WK_C_API_DEPRECATED;

WK_EXPORT WKStringRef WKFrameCopyMIMEType(WKFrameRef frame);
WK_EXPORT WKStringRef WKFrameCopyTitle(WKFrameRef frame);

WK_EXPORT WKPageRef WKFrameGetPage(WKFrameRef frame);

WK_EXPORT WKCertificateInfoRef WKFrameGetCertificateInfo(WKFrameRef frame) WK_C_API_DEPRECATED;

WK_EXPORT bool WKFrameCanProvideSource(WKFrameRef frame);
WK_EXPORT bool WKFrameCanShowMIMEType(WKFrameRef frame, WKStringRef mimeType) WK_C_API_DEPRECATED;

WK_EXPORT bool WKFrameIsDisplayingStandaloneImageDocument(WKFrameRef frame);
WK_EXPORT bool WKFrameIsDisplayingMarkupDocument(WKFrameRef frame);

WK_EXPORT bool WKFrameIsFrameSet(WKFrameRef frame) WK_C_API_DEPRECATED;

WK_EXPORT WKFrameHandleRef WKFrameCreateFrameHandle(WKFrameRef frame);
WK_EXPORT WKFrameInfoRef WKFrameCreateFrameInfo(WKFrameRef frame) WK_C_API_DEPRECATED;

typedef void (*WKFrameGetResourceDataFunction)(WKDataRef data, WKErrorRef error, void* functionContext);
WK_EXPORT void WKFrameGetMainResourceData(WKFrameRef frame, WKFrameGetResourceDataFunction function, void* functionContext);
WK_EXPORT void WKFrameGetResourceData(WKFrameRef frame, WKURLRef resourceURL, WKFrameGetResourceDataFunction function, void* functionContext);

typedef void (*WKFrameGetWebArchiveFunction)(WKDataRef archiveData, WKErrorRef error, void* functionContext);
WK_EXPORT void WKFrameGetWebArchive(WKFrameRef frame, WKFrameGetWebArchiveFunction function, void* functionContext);

#ifdef __cplusplus
}
#endif

#endif /* WKFrame_h */

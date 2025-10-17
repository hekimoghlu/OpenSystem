/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#ifndef WKBundleHitTestResult_h
#define WKBundleHitTestResult_h

#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKBundleHitTestResultMediaTypeNone,
    kWKBundleHitTestResultMediaTypeAudio,
    kWKBundleHitTestResultMediaTypeVideo
};
typedef uint32_t WKBundleHitTestResultMediaType;

WK_EXPORT WKTypeID WKBundleHitTestResultGetTypeID();

WK_EXPORT WKBundleNodeHandleRef WKBundleHitTestResultCopyNodeHandle(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKBundleNodeHandleRef WKBundleHitTestResultCopyURLElementHandle(WKBundleHitTestResultRef hitTestResult);

WK_EXPORT WKBundleFrameRef WKBundleHitTestResultGetFrame(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKBundleFrameRef WKBundleHitTestResultGetTargetFrame(WKBundleHitTestResultRef hitTestResult);

WK_EXPORT WKURLRef WKBundleHitTestResultCopyAbsoluteImageURL(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKURLRef WKBundleHitTestResultCopyAbsolutePDFURL(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKURLRef WKBundleHitTestResultCopyAbsoluteLinkURL(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKURLRef WKBundleHitTestResultCopyAbsoluteMediaURL(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT bool WKBundleHitTestResultMediaIsInFullscreen(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT bool WKBundleHitTestResultMediaHasAudio(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT bool WKBundleHitTestResultIsDownloadableMedia(WKBundleHitTestResultRef hitTestResultRef);
WK_EXPORT WKBundleHitTestResultMediaType WKBundleHitTestResultGetMediaType(WKBundleHitTestResultRef hitTestResult);

WK_EXPORT WKRect WKBundleHitTestResultGetImageRect(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKImageRef WKBundleHitTestResultCopyImage(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT bool WKBundleHitTestResultGetIsSelected(WKBundleHitTestResultRef hitTestResult);

WK_EXPORT WKStringRef WKBundleHitTestResultCopyLinkLabel(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKStringRef WKBundleHitTestResultCopyLinkTitle(WKBundleHitTestResultRef hitTestResult);
WK_EXPORT WKStringRef WKBundleHitTestResultCopyLinkSuggestedFilename(WKBundleHitTestResultRef hitTestResult);

#ifdef __cplusplus
}
#endif

#endif /* WKBundleHitTestResult_h */

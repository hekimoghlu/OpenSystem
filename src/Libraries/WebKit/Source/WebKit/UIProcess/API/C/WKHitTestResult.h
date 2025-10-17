/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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
#ifndef WKHitTestResult_h
#define WKHitTestResult_h

#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKHitTestResultGetTypeID(void);

WK_EXPORT WKURLRef WKHitTestResultCopyAbsoluteImageURL(WKHitTestResultRef hitTestResult);
WK_EXPORT WKURLRef WKHitTestResultCopyAbsolutePDFURL(WKHitTestResultRef hitTestResult);
WK_EXPORT WKURLRef WKHitTestResultCopyAbsoluteLinkURL(WKHitTestResultRef hitTestResult);
WK_EXPORT WKURLRef WKHitTestResultCopyAbsoluteMediaURL(WKHitTestResultRef hitTestResult);

WK_EXPORT WKStringRef WKHitTestResultCopyLinkLabel(WKHitTestResultRef hitTestResult);
WK_EXPORT WKStringRef WKHitTestResultCopyLinkTitle(WKHitTestResultRef hitTestResult);
WK_EXPORT WKStringRef WKHitTestResultCopyLookupText(WKHitTestResultRef hitTestResult);

WK_EXPORT bool WKHitTestResultIsContentEditable(WKHitTestResultRef hitTestResult);

WK_EXPORT WKRect WKHitTestResultGetElementBoundingBox(WKHitTestResultRef hitTestResultRef);

#ifdef __cplusplus
}
#endif

#endif /* WKHitTestResult_h */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef WKWebArchiveRef_h
#define WKWebArchiveRef_h

#include <WebKit/WKBase.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKWebArchiveGetTypeID();

WK_EXPORT WKWebArchiveRef WKWebArchiveCreate(WKWebArchiveResourceRef mainResource, WKArrayRef subresources, WKArrayRef subframeArchives);
WK_EXPORT WKWebArchiveRef WKWebArchiveCreateWithData(WKDataRef data);
WK_EXPORT WKWebArchiveRef WKWebArchiveCreateFromRange(WKBundleRangeHandleRef range);

WK_EXPORT WKWebArchiveResourceRef WKWebArchiveCopyMainResource(WKWebArchiveRef webArchive);
WK_EXPORT WKArrayRef WKWebArchiveCopySubresources(WKWebArchiveRef webArchive);
WK_EXPORT WKArrayRef WKWebArchiveCopySubframeArchives(WKWebArchiveRef webArchive);
WK_EXPORT WKDataRef WKWebArchiveCopyData(WKWebArchiveRef webArchive);

#ifdef __cplusplus
}
#endif

#endif // WKWebArchiveRef_h

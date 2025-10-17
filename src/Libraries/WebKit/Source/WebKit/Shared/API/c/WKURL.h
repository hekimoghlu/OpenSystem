/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#ifndef WKURL_h
#define WKURL_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKURLGetTypeID(void);

WK_EXPORT WKURLRef WKURLCreateWithUTF8CString(const char* string);
WK_EXPORT WKURLRef WKURLCreateWithUTF8String(const char* string, size_t length);
WK_EXPORT WKURLRef WKURLCreateWithBaseURL(WKURLRef baseURL, const char* relative);

WK_EXPORT WKStringRef WKURLCopyString(WKURLRef url);
WK_EXPORT WKStringRef WKURLCopyHostName(WKURLRef url);
WK_EXPORT WKStringRef WKURLCopyScheme(WKURLRef url);
WK_EXPORT WKStringRef WKURLCopyPath(WKURLRef url);
WK_EXPORT WKStringRef WKURLCopyLastPathComponent(WKURLRef url);

WK_EXPORT bool WKURLIsEqual(WKURLRef a, WKURLRef b);

#ifdef __cplusplus
}
#endif

#endif /* WKURL_h */

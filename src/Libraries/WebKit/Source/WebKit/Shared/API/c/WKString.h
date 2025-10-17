/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#ifndef WKString_h
#define WKString_h

#include <WebKit/WKBase.h>
#include <stddef.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(WIN32) && !defined(_WIN32) \
    && !((defined(__CC_ARM) || defined(__ARMCC__)) && !defined(__linux__)) /* RVCT */
    typedef unsigned short WKChar;
#else
    typedef wchar_t WKChar;
#endif

WK_EXPORT WKTypeID WKStringGetTypeID(void);

WK_EXPORT WKStringRef WKStringCreateWithUTF8CString(const char* string);
WK_EXPORT WKStringRef WKStringCreateWithUTF8CStringWithLength(const char* string, size_t stringLength);

WK_EXPORT bool WKStringIsEmpty(WKStringRef string);

WK_EXPORT size_t WKStringGetLength(WKStringRef string);
WK_EXPORT size_t WKStringGetCharacters(WKStringRef string, WKChar* buffer, size_t bufferLength);

WK_EXPORT size_t WKStringGetMaximumUTF8CStringSize(WKStringRef string);
WK_EXPORT size_t WKStringGetUTF8CString(WKStringRef string, char* buffer, size_t bufferSize);
WK_EXPORT size_t WKStringGetUTF8CStringNonStrict(WKStringRef string, char* buffer, size_t bufferSize);

WK_EXPORT bool WKStringIsEqual(WKStringRef a, WKStringRef b);
WK_EXPORT bool WKStringIsEqualToUTF8CString(WKStringRef a, const char* b);
WK_EXPORT bool WKStringIsEqualToUTF8CStringIgnoringCase(WKStringRef a, const char* b);

#ifdef __cplusplus
}
#endif

#endif /* WKString_h */

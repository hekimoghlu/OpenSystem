/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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

#ifndef HEADER_CURL_MULTIBYTE_H
#define HEADER_CURL_MULTIBYTE_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/
#include "curl_setup.h"

#if defined(_WIN32)

 /*
  * MultiByte conversions using Windows kernel32 library.
  */

wchar_t *curlx_convert_UTF8_to_wchar(const char *str_utf8);
char *curlx_convert_wchar_to_UTF8(const wchar_t *str_w);
#endif /* _WIN32 */

/*
 * Macros curlx_convert_UTF8_to_tchar(), curlx_convert_tchar_to_UTF8()
 * and curlx_unicodefree() main purpose is to minimize the number of
 * preprocessor conditional directives needed by code using these
 * to differentiate UNICODE from non-UNICODE builds.
 *
 * In the case of a non-UNICODE build the tchar strings are char strings that
 * are duplicated via strdup and remain in whatever the passed in encoding is,
 * which is assumed to be UTF-8 but may be other encoding. Therefore the
 * significance of the conversion functions is primarily for UNICODE builds.
 *
 * Allocated memory should be free'd with curlx_unicodefree().
 *
 * Note: Because these are curlx functions their memory usage is not tracked
 * by the curl memory tracker memdebug. You'll notice that curlx function-like
 * macros call free and strdup in parentheses, eg (strdup)(ptr), and that's to
 * ensure that the curl memdebug override macros do not replace them.
 */

#if defined(UNICODE) && defined(_WIN32)

#define curlx_convert_UTF8_to_tchar(ptr) curlx_convert_UTF8_to_wchar((ptr))
#define curlx_convert_tchar_to_UTF8(ptr) curlx_convert_wchar_to_UTF8((ptr))

typedef union {
  unsigned short       *tchar_ptr;
  const unsigned short *const_tchar_ptr;
  unsigned short       *tbyte_ptr;
  const unsigned short *const_tbyte_ptr;
} xcharp_u;

#else

#define curlx_convert_UTF8_to_tchar(ptr) (strdup)(ptr)
#define curlx_convert_tchar_to_UTF8(ptr) (strdup)(ptr)

typedef union {
  char                *tchar_ptr;
  const char          *const_tchar_ptr;
  unsigned char       *tbyte_ptr;
  const unsigned char *const_tbyte_ptr;
} xcharp_u;

#endif /* UNICODE && _WIN32 */

#define curlx_unicodefree(ptr)                          \
  do {                                                  \
    if(ptr) {                                           \
      (free)(ptr);                                      \
      (ptr) = NULL;                                     \
    }                                                   \
  } while(0)

#endif /* HEADER_CURL_MULTIBYTE_H */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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

#ifndef CURLINC_MPRINTF_H
#define CURLINC_MPRINTF_H
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

#include <stdarg.h>
#include <stdio.h> /* needed for FILE */
#include "curl.h"  /* for CURL_EXTERN */

#ifdef  __cplusplus
extern "C" {
#endif

#if (defined(__GNUC__) || defined(__clang__)) &&                        \
  defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) &&         \
  !defined(CURL_NO_FMT_CHECKS)
#if defined(__MINGW32__) && !defined(__clang__)
#define CURL_TEMP_PRINTF(fmt, arg) \
  __attribute__((format(gnu_printf, fmt, arg)))
#else
#define CURL_TEMP_PRINTF(fmt, arg) \
  __attribute__((format(printf, fmt, arg)))
#endif
#else
#define CURL_TEMP_PRINTF(fmt, arg)
#endif

CURL_EXTERN int curl_mprintf(const char *format, ...)
  CURL_TEMP_PRINTF(1, 2);
CURL_EXTERN int curl_mfprintf(FILE *fd, const char *format, ...)
  CURL_TEMP_PRINTF(2, 3);
CURL_EXTERN int curl_msprintf(char *buffer, const char *format, ...)
  CURL_TEMP_PRINTF(2, 3);
CURL_EXTERN int curl_msnprintf(char *buffer, size_t maxlength,
                               const char *format, ...)
  CURL_TEMP_PRINTF(3, 4);
CURL_EXTERN int curl_mvprintf(const char *format, va_list args)
  CURL_TEMP_PRINTF(1, 0);
CURL_EXTERN int curl_mvfprintf(FILE *fd, const char *format, va_list args)
  CURL_TEMP_PRINTF(2, 0);
CURL_EXTERN int curl_mvsprintf(char *buffer, const char *format, va_list args)
  CURL_TEMP_PRINTF(2, 0);
CURL_EXTERN int curl_mvsnprintf(char *buffer, size_t maxlength,
                                const char *format, va_list args)
  CURL_TEMP_PRINTF(3, 0);
CURL_EXTERN char *curl_maprintf(const char *format, ...)
  CURL_TEMP_PRINTF(1, 2);
CURL_EXTERN char *curl_mvaprintf(const char *format, va_list args)
  CURL_TEMP_PRINTF(1, 0);

#undef CURL_TEMP_PRINTF

#ifdef  __cplusplus
} /* end of extern "C" */
#endif

#endif /* CURLINC_MPRINTF_H */

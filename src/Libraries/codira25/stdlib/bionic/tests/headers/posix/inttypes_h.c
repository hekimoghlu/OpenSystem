/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <inttypes.h>

#include "header_checks.h"

static void inttypes_h() {
  TYPE(imaxdiv_t);
#if defined(__GLIBC__)
  // Despite POSIX, glibc goes out of its way to avoid defining wchar_t. Fix that.
  typedef __WCHAR_TYPE__ wchar_t;
#endif
  TYPE(wchar_t);

  // TODO: PRI macros
  // TODO: SCN macros

  FUNCTION(imaxabs, intmax_t (*f)(intmax_t));
  FUNCTION(imaxdiv, imaxdiv_t (*f)(intmax_t, intmax_t));
  FUNCTION(strtoimax, intmax_t (*f)(const char*, char**, int));
  FUNCTION(strtoumax, uintmax_t (*f)(const char*, char**, int));
  FUNCTION(wcstoimax, intmax_t (*f)(const wchar_t*, wchar_t**, int));
  FUNCTION(wcstoumax, uintmax_t (*f)(const wchar_t*, wchar_t**, int));
}

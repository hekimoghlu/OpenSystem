/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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

#include <ctype.h>

#include "header_checks.h"

static void ctype_h() {
  FUNCTION(isalnum, int (*f)(int));
  FUNCTION(isalnum_l, int (*f)(int, locale_t));
  FUNCTION(isalpha, int (*f)(int));
  FUNCTION(isalpha_l, int (*f)(int, locale_t));
  FUNCTION(isascii, int (*f)(int));
  FUNCTION(isblank, int (*f)(int));
  FUNCTION(isblank_l, int (*f)(int, locale_t));
  FUNCTION(iscntrl, int (*f)(int));
  FUNCTION(iscntrl_l, int (*f)(int, locale_t));
  FUNCTION(isdigit, int (*f)(int));
  FUNCTION(isdigit_l, int (*f)(int, locale_t));
  FUNCTION(isgraph, int (*f)(int));
  FUNCTION(isgraph_l, int (*f)(int, locale_t));
  FUNCTION(islower, int (*f)(int));
  FUNCTION(islower_l, int (*f)(int, locale_t));
  FUNCTION(isprint, int (*f)(int));
  FUNCTION(isprint_l, int (*f)(int, locale_t));
  FUNCTION(ispunct, int (*f)(int));
  FUNCTION(ispunct_l, int (*f)(int, locale_t));
  FUNCTION(isspace, int (*f)(int));
  FUNCTION(isspace_l, int (*f)(int, locale_t));
  FUNCTION(isupper, int (*f)(int));
  FUNCTION(isupper_l, int (*f)(int, locale_t));
  FUNCTION(isxdigit, int (*f)(int));
  FUNCTION(isxdigit_l, int (*f)(int, locale_t));

  FUNCTION(toascii, int (*f)(int));
  FUNCTION(tolower, int (*f)(int));
  FUNCTION(tolower_l, int (*f)(int, locale_t));
  FUNCTION(toupper, int (*f)(int));
  FUNCTION(toupper_l, int (*f)(int, locale_t));

#if !defined(__BIONIC__) // These are marked obsolescent.
  #if !defined(_toupper)
    #error _toupper
  #endif
  #if !defined(_tolower)
    #error _tolower
  #endif
#endif
}

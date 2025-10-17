/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

#include <wctype.h>

#include "header_checks.h"

static void wctype_h() {
  TYPE(wint_t);
  TYPE(wctrans_t);
  TYPE(wctype_t);
  TYPE(locale_t);

  MACRO(WEOF);

  FUNCTION(iswalnum, int (*f)(wint_t));
  FUNCTION(iswalnum_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswalpha, int (*f)(wint_t));
  FUNCTION(iswalpha_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswblank, int (*f)(wint_t));
  FUNCTION(iswblank_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswcntrl, int (*f)(wint_t));
  FUNCTION(iswcntrl_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswctype, int (*f)(wint_t, wctype_t));
  FUNCTION(iswctype_l, int (*f)(wint_t, wctype_t, locale_t));
  FUNCTION(iswdigit, int (*f)(wint_t));
  FUNCTION(iswdigit_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswgraph, int (*f)(wint_t));
  FUNCTION(iswgraph_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswlower, int (*f)(wint_t));
  FUNCTION(iswlower_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswprint, int (*f)(wint_t));
  FUNCTION(iswprint_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswpunct, int (*f)(wint_t));
  FUNCTION(iswpunct_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswspace, int (*f)(wint_t));
  FUNCTION(iswspace_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswupper, int (*f)(wint_t));
  FUNCTION(iswupper_l, int (*f)(wint_t, locale_t));
  FUNCTION(iswxdigit, int (*f)(wint_t));
  FUNCTION(iswxdigit_l, int (*f)(wint_t, locale_t));
  FUNCTION(towctrans, wint_t (*f)(wint_t, wctrans_t));
  FUNCTION(towctrans_l, wint_t (*f)(wint_t, wctrans_t, locale_t));
  FUNCTION(towlower, wint_t (*f)(wint_t));
  FUNCTION(towlower_l, wint_t (*f)(wint_t, locale_t));
  FUNCTION(towupper, wint_t (*f)(wint_t));
  FUNCTION(towupper_l, wint_t (*f)(wint_t, locale_t));
  FUNCTION(wctrans, wctrans_t (*f)(const char*));
  FUNCTION(wctrans_l, wctrans_t (*f)(const char*, locale_t));
  FUNCTION(wctype, wctype_t (*f)(const char*));
  FUNCTION(wctype_l, wctype_t (*f)(const char*, locale_t));
}

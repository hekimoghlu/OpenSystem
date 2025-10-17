/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

#include <locale.h>

#include "header_checks.h"

static void locale_h() {
  TYPE(struct lconv);
  STRUCT_MEMBER(struct lconv, char*, currency_symbol);
  STRUCT_MEMBER(struct lconv, char*, decimal_point);
  STRUCT_MEMBER(struct lconv, char, frac_digits);
  STRUCT_MEMBER(struct lconv, char*, grouping);
  STRUCT_MEMBER(struct lconv, char*, int_curr_symbol);
  STRUCT_MEMBER(struct lconv, char, int_frac_digits);
  STRUCT_MEMBER(struct lconv, char, int_n_cs_precedes);
  STRUCT_MEMBER(struct lconv, char, int_n_sep_by_space);
  STRUCT_MEMBER(struct lconv, char, int_n_sign_posn);
  STRUCT_MEMBER(struct lconv, char, int_p_cs_precedes);
  STRUCT_MEMBER(struct lconv, char, int_p_sep_by_space);
  STRUCT_MEMBER(struct lconv, char, int_p_sign_posn);
  STRUCT_MEMBER(struct lconv, char*, mon_decimal_point);
  STRUCT_MEMBER(struct lconv, char*, mon_grouping);
  STRUCT_MEMBER(struct lconv, char*, mon_thousands_sep);
  STRUCT_MEMBER(struct lconv, char*, negative_sign);
  STRUCT_MEMBER(struct lconv, char, n_cs_precedes);
  STRUCT_MEMBER(struct lconv, char, n_sep_by_space);
  STRUCT_MEMBER(struct lconv, char, n_sign_posn);
  STRUCT_MEMBER(struct lconv, char*, positive_sign);
  STRUCT_MEMBER(struct lconv, char, p_cs_precedes);
  STRUCT_MEMBER(struct lconv, char, p_sep_by_space);
  STRUCT_MEMBER(struct lconv, char, p_sign_posn);
  STRUCT_MEMBER(struct lconv, char*, thousands_sep);

  MACRO(NULL);

  MACRO(LC_ALL);
  MACRO(LC_COLLATE);
  MACRO(LC_CTYPE);
  MACRO(LC_MONETARY);
  MACRO(LC_NUMERIC);
  MACRO(LC_TIME);

  MACRO(LC_COLLATE_MASK);
  MACRO(LC_CTYPE_MASK);
  MACRO(LC_MESSAGES_MASK);
  MACRO(LC_MONETARY_MASK);
  MACRO(LC_NUMERIC_MASK);
  MACRO(LC_TIME_MASK);
  MACRO(LC_ALL_MASK);

  MACRO_TYPE(locale_t, LC_GLOBAL_LOCALE);
  TYPE(locale_t);

  FUNCTION(duplocale, locale_t (*f)(locale_t));
  FUNCTION(freelocale, void (*f)(locale_t));
  FUNCTION(localeconv, struct lconv* (*f)(void));
  FUNCTION(newlocale, locale_t (*f)(int, const char*, locale_t));
  FUNCTION(setlocale, char* (*f)(int, const char*));
  FUNCTION(uselocale, locale_t (*f)(locale_t));
}

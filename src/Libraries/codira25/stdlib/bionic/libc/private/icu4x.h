/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#pragma once

#include <ctype.h>
#include <stdint.h>
#include <wchar.h>

enum UCharCategory {
  U_NON_SPACING_MARK = 6,
  U_ENCLOSING_MARK = 7,
  U_DECIMAL_NUMBER = 9,
  U_CONTROL_CHAR = 15,
  U_FORMAT_CHAR = 16,
  U_DASH_PUNCTUATION = 19,
  U_OTHER_PUNCTUATION = 23,
};

enum UEastAsianWidth {
  U_EA_NEUTRAL,
  U_EA_AMBIGUOUS,
  U_EA_HALFWIDTH,
  U_EA_FULLWIDTH,
  U_EA_NARROW,
  U_EA_WIDE,
};

enum UHangulSyllableType {
  U_HST_NOT_APPLICABLE,
  U_HST_LEADING_JAMO,
  U_HST_VOWEL_JAMO,
  U_HST_TRAILING_JAMO,
  U_HST_LV_SYLLABLE,
  U_HST_LVT_SYLLABLE,
};

__BEGIN_DECLS

uint8_t __icu4x_bionic_general_category(uint32_t cp);
uint8_t __icu4x_bionic_east_asian_width(uint32_t cp);
uint8_t __icu4x_bionic_hangul_syllable_type(uint32_t cp);

bool __icu4x_bionic_is_alphabetic(uint32_t cp);
bool __icu4x_bionic_is_default_ignorable_code_point(uint32_t cp);
bool __icu4x_bionic_is_lowercase(uint32_t cp);
bool __icu4x_bionic_is_alnum(uint32_t cp);
bool __icu4x_bionic_is_blank(uint32_t cp);
bool __icu4x_bionic_is_graph(uint32_t cp);
bool __icu4x_bionic_is_print(uint32_t cp);
bool __icu4x_bionic_is_xdigit(uint32_t cp);
bool __icu4x_bionic_is_white_space(uint32_t cp);
bool __icu4x_bionic_is_uppercase(uint32_t cp);

uint32_t __icu4x_bionic_to_upper(uint32_t ch);
uint32_t __icu4x_bionic_to_lower(uint32_t ch);

__END_DECLS

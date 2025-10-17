/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#include <wchar.h>

#include "private/icu4x.h"

int wcwidth(wchar_t wc) {
  // Fast-path ASCII.
  if (wc >= 0x20 && wc < 0x7f) return 1;

  // ASCII NUL is a special case.
  if (wc == 0) return 0;

  // C0.
  if (wc < ' ' || (wc >= 0x7f && wc <= 0xa0)) return -1;

  // Now for the i18n part. This isn't defined or standardized, so a lot of the choices are
  // pretty arbitrary. See https://www.cl.cam.ac.uk/~mgk25/ucs/wcwidth.c for more details.

  // Fancy unicode control characters?
  switch (__icu4x_bionic_general_category(wc)) {
    case U_CONTROL_CHAR:
      return -1;
    case U_NON_SPACING_MARK:
    case U_ENCLOSING_MARK:
      return 0;
    case U_FORMAT_CHAR:
      // A special case for soft hyphen (U+00AD) to match historical practice.
      // See the tests for more commentary.
      return (wc == 0x00ad) ? 1 : 0;
  }

  // Medial and final jamo render as zero width when used correctly,
  // so we handle them specially rather than relying on East Asian Width.
  switch (__icu4x_bionic_hangul_syllable_type(wc)) {
    case U_HST_VOWEL_JAMO:
    case U_HST_TRAILING_JAMO:
      return 0;
    case U_HST_LEADING_JAMO:
    case U_HST_LV_SYLLABLE:
    case U_HST_LVT_SYLLABLE:
      return 2;
  }

  // Hangeul choseong filler U+115F is default ignorable, so we check default
  // ignorability only after we've already handled Hangeul jamo above.
  if (__icu4x_bionic_is_default_ignorable_code_point(wc)) return 0;

  // A few weird special cases where EastAsianWidth is not helpful for us.
  if (wc >= 0x3248 && wc <= 0x4dff) {
    // Circled two-digit CJK "speed sign" numbers. EastAsianWidth is ambiguous,
    // but wide makes more sense.
    if (wc <= 0x324f) return 2;
    // Hexagrams. EastAsianWidth is neutral, but wide seems better.
    if (wc >= 0x4dc0) return 2;
  }

  // The EastAsianWidth property is at least defined by the Unicode standard!
  // https://www.unicode.org/reports/tr11/
  switch (__icu4x_bionic_east_asian_width(wc)) {
    case U_EA_AMBIGUOUS:
    case U_EA_HALFWIDTH:
    case U_EA_NARROW:
    case U_EA_NEUTRAL:
      return 1;
    case U_EA_FULLWIDTH:
    case U_EA_WIDE:
      return 2;
  }

  return 0;
}
